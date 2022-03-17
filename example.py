from collections import OrderedDict, defaultdict

from docarray import Document, DocumentArray
from jina import Client

client = Client(protocol='grpc', port='45678')


def predict_tags(doc: 'Document', matches: 'DocumentArray'):
    doc_caption = doc.tags.get('caption', '')
    doc_tags = set(doc_caption.split(' '))
    token_weights = defaultdict(float)
    token_counts = defaultdict(int)
    for m in matches:
        # convert distance to similarity score
        sim_score = 1 - m.scores['cosine'].value

        m_caption = m.tags.get('caption', '')
        for t in set(m_caption.split(' ')):
            if t in doc_tags:
                continue
            token_counts[t] += 1
            token_weights[t] += sim_score

    for k, c in token_counts.items():
        token_weights[k] = token_weights[k] / c

    return OrderedDict(sorted(token_weights.items(), key=lambda x: x[1], reverse=True))


examples = DocumentArray.load_binary('../text-image-retrieval/data/sample_da_12')
for doc in examples[150:]:
    # conduct image document
    if not doc.blob and len(doc.chunks) > 0:
        doc.blob = doc.chunks.sample(1)[0].blob

        # # optional: to save image for manual check
        # doc.save_blob_to_file('example.jpg')

    print(f'==> input doc: {doc.tags["caption"]}')

    result = client.post('/search', doc, return_results=True)[0]
    predicts = predict_tags(doc, result.matches)
    print(f'==> predicts: {predicts}')
    # TODO: select the TOP-K from predicts

    # input('Enter to continue ...')
    break
