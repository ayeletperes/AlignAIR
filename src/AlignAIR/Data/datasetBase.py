import numpy as np
import csv
from GenAIRR.dataconfig import DataConfig
import tensorflow as tf
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from ast import literal_eval  # safer than eval for parsing literal structures

from .columnSet import ColumnSet
from .tokenizers import CenterPaddedSequenceTokenizer, CenterPaddedAminoAcidTokenizer
from .encoders import AlleleEncoder
from .batch_readers import StreamingTableReader

class DatasetBase(ABC):
    """
    Base dataset class for AIRR modeling.

    Provides infrastructure for tokenizing sequences, encoding alleles, and reading data
    in either full or streaming mode. Subclasses must implement dictionary derivation
    and batch assembly logic.
    """
    def __init__(self, data_path, dataconfig: DataConfig, batch_size=64, max_sequence_length=512, use_streaming=False,
                 nrows=None, seperator=',',required_data_columns=None,
                 max_aa_sequence_length=None, use_aa_stream=False):

        self.v_dict = None
        self.d_dict = None
        self.j_dict = None

        self.max_sequence_length = max_sequence_length
        self.use_aa_stream = use_aa_stream

        self.dataconfig = dataconfig
        self.tokenizer = CenterPaddedSequenceTokenizer(max_length=max_sequence_length)
        if self.use_aa_stream:
            self.max_sequence_length = max_aa_sequence_length or (max_sequence_length // 3)
            self.tokenizer = CenterPaddedAminoAcidTokenizer(max_length=self.max_sequence_length)
        self.allele_encoder = AlleleEncoder()

        self.seperator = seperator
        self.use_streaming  = use_streaming
        self.required_data_columns = ColumnSet() if required_data_columns is None else required_data_columns
        self.batch_size = batch_size
        self.add_allele_dictionaries()
        self.register_alleles_to_ohe()

        # AA-mode only: precompute AA-equivalence groups per gene so that
        # ground-truth allele targets can be expanded to multi-hot over
        # all AA-identical alleles. The model cannot distinguish alleles
        # within an AA group, so penalizing it for picking the "wrong"
        # one wastes loss signal.
        self.aa_equiv = None
        if self.use_aa_stream:
            from GenAIRR.utilities.aa_equivalence import aa_equivalent_alleles
            # MultiDataConfigContainer holds a list of dataconfigs; for
            # single-chain training there is exactly one.
            dc = self.dataconfig.dataconfigs[0] if hasattr(self.dataconfig, 'dataconfigs') else self.dataconfig
            self.aa_equiv = {'V': aa_equivalent_alleles(dc, 'v'),
                             'J': aa_equivalent_alleles(dc, 'j')}
            if self.has_d:
                self.aa_equiv['D'] = aa_equivalent_alleles(dc, 'd')

        self.data_path = data_path

        # Use unified reader: stream=True for streaming mode, False for in-memory mode
        self.reader = StreamingTableReader(
            data_path,
            self.required_data_columns,
            batch_size,
            sep=seperator,
            stream=use_streaming,
        )

        self.data_length = self.reader.get_data_length()

    def register_alleles_to_ohe(self):
        """
        Register alleles to the one-hot encoder based on the derived dictionaries.
        Returns:

        """
        v_alleles = sorted(list(self.v_dict))
        j_alleles = sorted(list(self.j_dict))

        self.v_allele_count = len(v_alleles)
        self.j_allele_count = len(j_alleles)

        self.allele_encoder.register_gene("V", v_alleles, sort=False)
        self.allele_encoder.register_gene("J", j_alleles, sort=False)

        if self.has_d:
            d_alleles = sorted(list(self.d_dict)) + ['Short-D']
            self.d_allele_count = len(d_alleles)
            self.allele_encoder.register_gene("D", d_alleles, sort=False)


    @property
    def has_d(self):
        return self.dataconfig.metadata.has_d

    def add_allele_dictionaries(self):
        self.v_dict = {i.name:i for i in self.dataconfig.allele_list('v')}
        self.j_dict = {i.name:i for i in self.dataconfig.allele_list('j')}

        if self.has_d:
            self.d_dict = {i.name:i for i in self.dataconfig.allele_list('d')}

    def encode_and_equal_pad_sequence(self, sequence):
        return self.tokenizer.encode_and_pad_center(sequence)

    def _expand_aa_equiv(self, allele_set: set, gene: str) -> set:
        """Expand each allele in ``allele_set`` to its AA-equivalence group.

        Only meaningful in AA mode; ``self.aa_equiv`` is None in nt mode.
        Alleles not present in the equivalence map (unexpected) pass
        through unchanged.
        """
        mapping = self.aa_equiv[gene]
        expanded = set()
        for a in allele_set:
            if a in mapping:
                expanded |= mapping[a]
            else:
                expanded.add(a)
        return expanded

    def get_ohe_reverse_mapping(self):
        return self.allele_encoder.get_reverse_mapping()

    def one_hot_encode_allele(self, gene, ground_truth_labels):
        return self.allele_encoder.encode(gene, ground_truth_labels)

    @property
    def _loaded_genes(self):
        """
        return the list of loaded genes in the dataset with the context of the dataconfig.
        Returns:
            List[str]: List of gene names that are loaded in the dataset.
        """
        genes = ['v', 'j']
        if self.has_d:
            genes.append('d')
        return genes



    def _get_single_batch(self, pointer):
        """Assemble a single training batch without pandas.

        raw_batch structure (dict[str, list]): produced by BatchReader implementations.
        Required keys include: sequence, v_call, j_call, mutation_rate, productive, n_indels,
        and per-gene start/end coordinate columns (e.g. v_sequence_start).
        """
        batch = self.reader.get_batch(pointer)
          
        sequences = batch['sequence_aa'] if self.use_aa_stream else batch['sequence']
        encoded_sequences, paddings = self.tokenizer.encode_and_pad_center(sequences)
        pad_int = paddings.astype(np.int32)

        # Adjust gene coordinates in place. In AA mode the CSV's
        # *_sequence_start/end columns already hold AA-space indices
        # (the preprocess_aa_training_data script converts them).
        # In nt mode they hold nt-space indices. Either way, the pad
        # offset comes from the active tokenizer in matching units.
        for gene in self._loaded_genes:
            s_key = f'{gene}_sequence_start'
            e_key = f'{gene}_sequence_end'
            if s_key in batch:
                arr = np.asarray(batch[s_key], dtype=np.int32)
                batch[s_key] = (arr + pad_int).astype(np.float32)
            if e_key in batch:
                arr = np.asarray(batch[e_key], dtype=np.int32)
                batch[e_key] = (arr + pad_int).astype(np.float32)

        # Indel counts (later: prefer a precomputed indel_count column)
        indel_counts = self._parse_indel_counts(batch['n_indels'])

        # Allele sets. In AA mode, expand each ground-truth allele to
        # its AA-equivalence group (all alleles with identical translated
        # germline). The encoder already accepts a set per row.
        v_alleles = [set(s.split(',')) for s in batch['v_call']]
        j_alleles = [set(s.split(',')) for s in batch['j_call']]
        if self.use_aa_stream:
            v_alleles = [self._expand_aa_equiv(s, 'V') for s in v_alleles]
            j_alleles = [self._expand_aa_equiv(s, 'J') for s in j_alleles]

        # Accept already-converted floats/bools for productive; fallback to string parsing
        def to_float_bool(v):
            if isinstance(v, bool):
                return 1.0 if v else 0.0
            if isinstance(v, (int, float)):
                return 1.0 if v != 0 else 0.0
            s = str(v).strip().lower()
            if s in {'true','1','yes','y','t'}:
                return 1.0
            if s in {'false','0','no','n','f'}:
                return 0.0
            return 0.0

        x = {'tokenized_sequence': encoded_sequences}

        y = {
            'v_start': np.asarray(batch['v_sequence_start'], np.float32).reshape(-1, 1),
            'v_end':   np.asarray(batch['v_sequence_end'],   np.float32).reshape(-1, 1),
            'j_start': np.asarray(batch['j_sequence_start'], np.float32).reshape(-1, 1),
            'j_end':   np.asarray(batch['j_sequence_end'],   np.float32).reshape(-1, 1),
            'v_allele': self.one_hot_encode_allele('V', v_alleles),
            'j_allele': self.one_hot_encode_allele('J', j_alleles),
            'mutation_rate': np.asarray(batch['mutation_rate'], np.float32).reshape(-1, 1),
            'indel_count': np.asarray(indel_counts, np.float32).reshape(-1, 1),
            'productive': np.asarray([to_float_bool(v) for v in batch['productive']], np.float32).reshape(-1, 1)
        }

        if self.has_d:
            d_alleles = [set(s.split(',')) for s in batch['d_call']]
            if self.use_aa_stream:
                d_alleles = [self._expand_aa_equiv(s, 'D') for s in d_alleles]
            y['d_allele'] = self.one_hot_encode_allele('D', d_alleles)
            y['d_start'] = np.asarray(batch['d_sequence_start'], np.float32).reshape(-1, 1)
            y['d_end']   = np.asarray(batch['d_sequence_end'],   np.float32).reshape(-1, 1)

        return x, y

    # ---- Helper Methods (pandas-free) --------------------------------------------------
    def _parse_indel_counts(self, indel_column):
        """Parse indel structures (dict or stringified dict) into counts.
        Fast path: dict -> len(dict). Fallback: literal_eval for strings.
        """
        counts = []
        for item in indel_column:
            # Fast path: already structured
            if isinstance(item, (dict, list, tuple)):
                counts.append(len(item))
                continue
            # Fallback: try parsing strings
            try:
                parsed = literal_eval(item) if isinstance(item, str) else {}
                counts.append(len(parsed) if isinstance(parsed, (dict, list, tuple)) else 0)
            except Exception:
                counts.append(0)
        return counts

    def _get_tf_dataset_params(self):

        x, y = self._get_single_batch(self.batch_size)

        output_types = ({k: tf.float32 for k in x}, {k: tf.float32 for k in y})

        output_shapes = ({k: (self.batch_size,) + x[k].shape[1:] for k in x},
                         {k: (self.batch_size,) + y[k].shape[1:] for k in y})

        return output_types, output_shapes

    def _train_generator(self):
        """Infinite generator over batches.

        For streaming readers the pointer is inconsequential (reader is stateful).
        For in-memory (Pandas) reader we still advance a pointer for wrap-around, but logic
        simplified for clarity.
        """
        pointer = 0
        while True:
            batch_x, batch_y = self._get_single_batch(pointer)
            pointer += self.batch_size
            if pointer >= self.data_length:
                pointer = 0
            # Guarantee full batch size (skip partials only in non-streaming edge cases)
            if len(batch_x['tokenized_sequence']) == self.batch_size:
                yield batch_x, batch_y

    def get_train_dataset(self):
        output_types, output_shapes = self._get_tf_dataset_params()

        dataset = tf.data.Dataset.from_generator(
            lambda: self._train_generator(),
            output_types=output_types,
            output_shapes=output_shapes,
        )
        # Add asynchronous prefetch to enable true producer/consumer behavior
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def generate_model_params(self):
        params = {
            "max_seq_length": self.max_sequence_length,
            'dataconfig': self.dataconfig,
        }
        # In AA mode, self.max_sequence_length was already overwritten to the
        # AA length in __init__, so the model picks up the right input length
        # from max_seq_length alone. We just need to flag the mode so the
        # model picks the AA vocab (23) instead of the nt vocab (6).
        if self.use_aa_stream:
            params['use_aa_stream'] = True
        return params

    def __len__(self):
        return self.data_length

