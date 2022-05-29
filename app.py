import logging
from bson import ObjectId
import uuid
import numpy as np
import pika
from environs import Env
from sklearn.cluster import DBSCAN
from scipy import sparse
from mongoengine import connect, disconnect
from imagenetviewer.image import Image, Neighbor, ImageStatus
from sklearn import preprocessing
from mongoengine.queryset.visitor import Q
from pymilvus import connections
from tqdm import tqdm


def on_request(ch, method, props, body):
    clustering_request_id = body.decode()

    # .only('clustering_request_id').only('neighbors')
    images = Image.objects(clustering_request_id=clustering_request_id)

    n_objects = len(images)

    logger.debug('encode object ids')
    images_ids = [image.id for image in images]
    le = preprocessing.LabelEncoder()
    le.fit(images_ids)

    logger.debug('build (sparse) distance matrix')
    row_ids = []
    col_ids = []
    dists = []
    for image in tqdm(images):
        image_num_id = le.transform([image.id])[0]
        for neighbor in image.neighbors:
            matched_image_num_id = le.transform([neighbor.matched_image.id])[0]

            row_ids.append(image_num_id)
            col_ids.append(matched_image_num_id)
            dists.append(neighbor.distance)

            row_ids.append(matched_image_num_id)
            col_ids.append(image_num_id)
            dists.append(neighbor.distance)

    dm = sparse.csr_matrix((dists, (row_ids, col_ids)), shape=(n_objects, n_objects))

    clustering = DBSCAN(
        eps=env.float('CLUSTERING_THRESHOLD'),
        min_samples=env.int('CLUSTERING_MIN_CLUSTER'),
        metric='precomputed'
    ).fit(dm)

    unique_cluster_labels = np.unique(clustering.labels_)
    cluster_label_to_id = {cluster_label: str(uuid.uuid4()) for cluster_label in unique_cluster_labels}

    logger.debug('writing results')
    for image in tqdm(images):
        image_num_id = le.transform([image.id])[0]
        cluster_label = clustering.labels_[image_num_id]
        cluster_id = cluster_label_to_id[cluster_label]

        if cluster_label == -1:
            del image.neighbors
            del image.clustering_request_id
            image.status = ImageStatus.PENDING_MATCHING
            image.save()

            ch.basic_publish(
                exchange='',
                routing_key=env('MATCHING_INPUT_QUEUE'),
                body=str(image.id)
            )
        else:
            image.cluster_id = cluster_id
            image.status = ImageStatus.CLUSTERIZED
            image.save()

    n_clusters = (len(unique_cluster_labels) - 1) if -1 in unique_cluster_labels else len(unique_cluster_labels)
    logger.info(f'found {n_clusters} clusters in {clustering_request_id}')

    ch.basic_ack(delivery_tag=method.delivery_tag)



if __name__ == '__main__':
    matching_processed_n = 0

    logger = logging.getLogger('clusterizer')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s %(name)s %(levelname)s] %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    env = Env()
    env.read_env()

    connect(host=env('MONGODB_HOST'), uuidRepresentation='standard')

    con_par = pika.ConnectionParameters(
        heartbeat=600,
        blocked_connection_timeout=300,
        host=env('RABBITMQ_HOST')
    )
    connection = pika.BlockingConnection(con_par)
    channel = connection.channel()

    channel.queue_declare(queue=env('INPUT_QUEUE'), durable=True)
    channel.queue_declare(queue=env('OUTPUT_QUEUE'), durable=True)
    channel.queue_declare(queue=env('MATCHING_INPUT_QUEUE'), durable=True)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=env('INPUT_QUEUE'), on_message_callback=on_request)

    logger.info('[+] awaiting images to recognize')
    channel.start_consuming()
