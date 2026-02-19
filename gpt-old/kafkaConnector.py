from confluent_kafka import Producer, Consumer
import json
import logging
import traceback
import os

# Set up the logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) 

class kafkaConnector():
    auto_offset_reset = 'earliest'
    enable_auto_commit = True
    group_id = None
    main_topic = None
    error_topic = None
    consumer = None
    reprocess_consumer = None
    timeout = 8
    max_poll_records = 128
    producer = None
    reprocess_producer = None
    retries = 5
    reprocess_time = 300
    repoll_time = 10
    max_attempts = 3
    config = {}

    def __init__(self, main_topic, error_topic, **kwargs):
        self.main_topic = main_topic
        self.error_topic = error_topic
        self.max_poll_records = int(os.getenv("BATCH_SIZE", "128"))
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def value_deserializer_consumer(x):
        return x.decode('utf-8')

    @staticmethod
    def value_serializer_producer(m):
        return json.dumps(m).encode('ascii')

    def initialize_consummer(self):
        consumer_config = self.config | {
            "group.id": self.group_id,
            "auto.offset.reset": self.auto_offset_reset
        }
        self.consumer = Consumer(consumer_config)

        self.consumer.subscribe(topics=[self.main_topic])
        return

    def initialize_reprocess_consummer(self):
        consumer_config = self.config | {
            "group.id": self.group_id,
            "auto.offset.reset": self.auto_offset_reset
        }
        self.reprocess_consumer = Consumer(consumer_config)

        self.reprocess_consumer.subscribe(topics=[self.error_topic])

    def get_messages(self, mode="main"):
        try:
            if mode == "main":
                if self.consumer==None:
                    self.initialize_consummer()
                consumer = self.consumer
            elif mode == "error":
                if self.reprocess_consumer==None:
                    self.initialize_reprocess_consummer()
                consumer = self.reprocess_consumer
            if consumer == None:
                raise TypeError("Consumer is not Initialize")
            msgs = consumer.consume(
                num_messages=self.max_poll_records, timeout=self.timeout)
            logger.info(msgs)
            print(msgs)
            msg_values = []
            if len(msgs)<1:
                return False, msg_values
            for message in msgs:
                try:
                    msg_value = message.value()
                    logger.info(msg_value)
                    if msg_value is None or len(msg_value.strip()) == 0:
                        logger.warning("Received empty message, skipping")
                        continue
                    decoded_msg = self.value_deserializer_consumer(msg_value)
                    logger.info(decoded_msg)
                    if decoded_msg is None or len(decoded_msg.strip()) == 0:
                        logger.warning("Decoded message is empty, skipping")
                        continue
                    parsed_msg = json.loads(decoded_msg)
                    logger.info(parsed_msg)
                    msg_values.append(parsed_msg)
                except json.JSONDecodeError as je:
                    logger.error(f"Failed to parse message as JSON: {je}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue
                    
            if len(msg_values) > 0:
                consumer.commit()
                return True, msg_values
            return False, msg_values
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"Error in get_messages: {tb_str}")
            return False, []

    def initialize_producer(self):
        self.producer = Producer(self.config)

    def initialize_reprocess_producer(self):
        self.reprocess_producer = Producer(self.config)

    def send_error_message(self, message):
        if self.reprocess_producer==None:
            self.initialize_reprocess_producer()
        self.reprocess_producer.produce(self.error_topic, value=self.value_serializer_producer(message))
        try:
            self.reprocess_producer.flush()
            return True
        except:
            return False

    def send_reprocess_message(self, message):
        if self.producer==None:
            self.initialize_producer()
        self.producer.produce(
            self.main_topic, value=self.value_serializer_producer(message))
        try:
            self.producer.flush()
            return True
        except:
            return False

    def reprocess_messages(self):
        try:
            # self.initialize_reprocess_consummer()

            while True:
                _, msgs = self.get_messages(mode="error")
                for msg in msgs:
                    print(msg)
                    if not msg:
                        return True

                    if "attempt" in msg.keys() and msg["attempt"] <= self.max_attempts:
                        self.send_reprocess_message(msg)
                        continue

                    response = self.send_error_message(msg)
                    if not response:
                        return False

        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"\nAn Error occured:\n{tb_str}")
            return False
