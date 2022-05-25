from typing import List, Tuple

from spacy.tokens import Doc, Span


class CorefResolver(object):
    """a class that implements the logic from
    https://towardsdatascience.com/how-to-make-an-effective-coreference-resolution-model-55875d2b5f19"""

    def __init__(self) -> None:
        pass

    @staticmethod
    def core_logic_part(document: Doc, coref: List[int], resolved: List[str], mention_span: Span):
        """
        If the last token of the mention is a possessive pronoun, then add an apostrophe and an s to the mention.
        Otherwise, just add the last token to the mention

        :param document: Doc object
        :type document: Doc
        :param coref: List[int]
        :param resolved: list of strings, where each string is a token in the sentence
        :param mention_span: The span of the mention that we want to replace
        :return: The resolved list is being returned.
        """
        final_token = document[coref[1]]
        final_token_tag = str(final_token.tag_).lower()
        test_token_test = False
        for option in ["PRP$", "POS", "BEZ"]:
            if option.lower() in final_token_tag:
                test_token_test = True
                break
        if test_token_test:
            resolved[coref[0]] = mention_span.text + "'s" + final_token.whitespace_
        else:
            resolved[coref[0]] = mention_span.text + final_token.whitespace_
        for i in range(coref[0] + 1, coref[1] + 1):
            resolved[i] = ""
        return resolved

    @staticmethod
    def get_span_noun_indices(doc: Doc, cluster: List[List[int]]) -> List[int]:
        """
        > Get the indices of the spans in the cluster that contain at least one noun or proper noun

        :param doc: Doc
        :param cluster: List[List[int]]
        :return: A list of indices of spans that contain at least one noun or proper noun.
        """
        spans = [doc[span[0] : span[1] + 1] for span in cluster]
        spans_pos = [[token.pos_ for token in span] for span in spans]
        span_noun_indices = [
            i for i, span_pos in enumerate(spans_pos) if any(pos in span_pos for pos in ["NOUN", "PROPN"])
        ]
        return span_noun_indices

    @staticmethod
    def get_cluster_head(doc: Doc, cluster: List[List[int]], noun_indices: List[int]):
        """
        > Given a spaCy Doc, a list of clusters, and a list of noun indices, return the head span and its start and end
        indices

        :param doc: the spaCy Doc object
        :type doc: Doc
        :param cluster: a list of lists, where each sublist is a span of tokens in the document
        :type cluster: List[List[int]]
        :param noun_indices: a list of indices of the nouns in the cluster
        :type noun_indices: List[int]
        :return: The head span and the start and end indices of the head span.
        """
        head_idx = noun_indices[0]
        head_start, head_end = cluster[head_idx]
        head_span = doc[head_start : head_end + 1]
        return head_span, [head_start, head_end]

    @staticmethod
    def is_containing_other_spans(span: List[int], all_spans: List[List[int]]):
        """
        It returns True if there is any span in all_spans that is contained within span and is not equal to span

        :param span: the span we're checking to see if it contains other spans
        :type span: List[int]
        :param all_spans: a list of all the spans in the document
        :type all_spans: List[List[int]]
        :return: A list of all spans that are not contained in any other span.
        """
        return any([s[0] >= span[0] and s[1] <= span[1] and s != span for s in all_spans])

    def replace_corefs(self, document: Doc, clusters: List[List[int]]) -> Tuple[str, dict]:
        """
        > For each cluster, find the head noun, and replace all other mentions with the head noun

        :param document: the spacy Doc object
        :type document: Doc
        :param clusters: list of lists of spans
        :type clusters: List[List[int]]
        :return: The resolved str is being returned.
        """
        resolved = list(tok.text_with_ws for tok in document)
        all_spans = [span for cluster in clusters for span in cluster]  # flattened list of all spans
        cluster_heads = {}
        for cluster in clusters:
            noun_indices = self.get_span_noun_indices(document, cluster)

            if noun_indices:
                mention_span, mention = self.get_cluster_head(document, cluster, noun_indices)
                cluster_heads[mention_span] = mention

                for coref in cluster:
                    if coref != mention and not self.is_containing_other_spans(coref, all_spans):
                        self.core_logic_part(document, coref, resolved, mention_span)
        return "".join(resolved), cluster_heads
