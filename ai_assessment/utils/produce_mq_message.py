import os
import json
import sys
from typing import Dict, Any
from mq_http_sdk.mq_exception import MQExceptionBase
from mq_http_sdk.mq_producer import *
from mq_http_sdk.mq_client import *


def init_producer():
    mq_client = MQClient(
        os.getenv("ROCKETMQ_HTTP_ENDPOINT"),
        os.getenv("ROCKETMQ_ACCESS_KEY"),
        os.getenv("ROCKETMQ_SECRET_KEY")
        )
    instance_id = os.getenv("ROCKETMQ_INSTANCE_ID")
    topic_name = os.getenv("ROCKETMQ_TOPIC_ID")
    group_id = os.getenv("ROCKETMQ_GROUP_ID")
    producer = mq_client.get_trans_producer(instance_id, topic_name, group_id)
    return producer


def produce_one_mq_message(json_message: Dict[str, Any]):
    try:
        producer = init_producer()
    except Exception as e:
        print("Producer Initialization Fail. Exception:%s" % e)
        return
    try:
        msg = TopicMessage(
            #message content
            json.dumps(json_message, ensure_ascii=False, indent=4),
            #message tag
            os.getenv("ROCEKTMQ_TAG")
            )
        re_msg = producer.publish_message(msg)
        print("Publish Message Succeed. MessageID:%s, BodyMD5:%s" % (re_msg.message_id, re_msg.message_body_md5))
    except MQExceptionBase as e:
        if e.type == "TopicNotExist":
            print("Topic not exist, please create it.")
            sys.exit(1)
        print("Publish Message Fail. Exception:%s" % e)
