import numpy as np


class CenterPaddedAminoAcidTokenizer:
    """
    Encodes and center-pads amino-acid sequences using a token dictionary.

    This tokenizer mirrors :class:`CenterPaddedSequenceTokenizer` for the AA
    stream, so the original AA sequence appears in the center of the final
    array with equal (or nearly equal) padding on both sides. Pairing AA
    samples with the nt tokenizer is then a matter of consistent center-padding
    on both sides.

    The default vocab covers the 20 standard amino acids (1-20), ``*`` for
    stop codons (21), ``X`` for ambiguous / invalid codons (22, as emitted by
    GenAIRR's translator for codons containing non-A/C/G/T bases), and ``_``
    for padding (0). ``_`` is used instead of an AA letter because every
    standard single-letter code is a real amino acid (e.g. ``P`` is Proline)
    and would collide with real sequence positions.

    Attributes:
        token_dict (dict): Mapping of amino-acid characters to integers.
        max_length (int): Maximum padded sequence length.
    """

    def __init__(self, token_dict=None, max_length=192):
        """
        Args:
            token_dict (dict, optional): Custom token mapping. Defaults to
                20 standard AAs + ``*`` (stop) + ``X`` (ambiguous) + ``_`` (pad).
            max_length (int): Desired padded sequence length. Typically
                ``nt_max_length // 3`` (e.g. 192 for an nt length of 576).
        """
        self.token_dict = token_dict or {
            "A": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "F": 5,
            "G": 6,
            "H": 7,
            "I": 8,
            "K": 9,
            "L": 10,
            "M": 11,
            "N": 12,
            "P": 13,
            "Q": 14,
            "R": 15,
            "S": 16,
            "T": 17,
            "V": 18,
            "W": 19,
            "Y": 20,
            "*": 21,
            "X": 22,
            "_": 0,  # padding token
        }
        self.max_length = max_length

    def encode(self, sequence):
        """
        Converts an amino-acid string to an encoded array of integers.

        Any character not present in :attr:`token_dict` (e.g. unexpected
        IUPAC ambiguity codes like ``B`` or ``Z``) is mapped to ``X``.

        Args:
            sequence (str): Input amino-acid string.

        Returns:
            np.ndarray: Encoded sequence.
        """
        return np.array([self.token_dict.get(aa, self.token_dict['X']) for aa in sequence])

    def encode_and_pad_center(self, sequence_or_sequences):
        """
        Encodes and center-pads one or multiple amino-acid sequences.

        Args:
            sequence_or_sequences (str or Iterable[str]):
                A single amino-acid string or an iterable of amino-acid strings.

        Returns:
            If input is a string:
                Tuple[np.ndarray, int]: (padded_sequence, left_padding)
            If input is an iterable:
                Tuple[np.ndarray, np.ndarray]: (batch_padded_sequences, paddings_array)
        """
        if isinstance(sequence_or_sequences, str):
            encoded = self.encode(sequence_or_sequences)
            padding_length = self.max_length - len(encoded)
            pad_left = padding_length // 2
            pad_right = padding_length - pad_left
            padded = np.pad(encoded, (pad_left, pad_right), constant_values=0)
            return padded, pad_left

        # If it's a list/iterable of sequences
        padded_batch = []
        paddings = []

        for seq in sequence_or_sequences:
            encoded = self.encode(seq)
            padding_length = self.max_length - len(encoded)
            pad_left = padding_length // 2
            pad_right = padding_length - pad_left
            padded = np.pad(encoded, (pad_left, pad_right), constant_values=0)
            padded_batch.append(padded)
            paddings.append(pad_left)

        return np.vstack(padded_batch), np.array(paddings)
