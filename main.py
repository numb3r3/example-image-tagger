__copyright__ = 'Copyright (c) 2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

import os

import click
from docarray import Document, DocumentArray
from jina import Flow

cur_dir = os.path.dirname(os.path.abspath(__file__))


@click.group()
def cli():
    ...


@cli.command()
@click.option('-p', '--port', default=45678, help='the gateway port')
@click.option('--protocol', default='grpc', help='the gateway protocol')
@click.option('-f', '--flow_path', help='the flow yaml file path')
def serve(flow_path, port, protocol):
    assert flow_path, f'Please specify the Flow path.'
    flow = Flow.load_config(flow_path)

    # setting the gateway protocol and port at the run time.
    flow.port = port
    flow.protocol = protocol

    flow.expose_endpoint('/status', summary='Return the status of indexer')
    flow.expose_endpoint('/clear', summary='Clear the indexer data')
    flow.expose_endpoint('/sync', summary='Sync the indexer data')
    with flow:
        flow.block()


@cli.command()
@click.option('-d', '--data_path', help='the dataset file path')
@click.option('-f', '--flow_path', help='the flow yaml file path')
@click.option('-l', '--limit', type=int, help='the maximum number of docs to index')
def index(flow_path, data_path, limit):
    assert flow_path, f'Please specify the Flow path.'
    flow = Flow.load_config(flow_path)
    with flow as f:
        print(f'==> loading dataset: {data_path} ...')
        da = DocumentArray.load_binary(data_path)

        def doc_generate():
            count = 0
            for doc in da:
                # image is located at root doc default.
                image = doc

                if len(doc.chunks) > 0:
                    # image are located at chunk-level doc
                    # random sample at each step
                    image = doc.chunks.sample(1)[0]
                    image.tags = doc.tags

                if not (image.tensor is not None or image.blob != b'' or image.uri):
                    continue

                count += 1
                yield image
                if limit and count >= limit:
                    break

        print(f'==> indexing to local folder at `./workspace` ...')
        f.post(on='/index', inputs=doc_generate(), request_size=64, show_progress=True)


if __name__ == '__main__':
    cli()
