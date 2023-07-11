# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: object_detection/protos/input_reader.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n*object_detection/protos/input_reader.proto\x12\x17object_detection.protos\"\xde\x04\n\x0bInputReader\x12\x18\n\x0elabel_map_path\x18\x01 \x01(\t:\x00\x12\x15\n\x07shuffle\x18\x02 \x01(\x08:\x04true\x12!\n\x13shuffle_buffer_size\x18\x0b \x01(\r:\x04\x32\x30\x34\x38\x12*\n\x1d\x66ilenames_shuffle_buffer_size\x18\x0c \x01(\r:\x03\x31\x30\x30\x12\x1c\n\x0equeue_capacity\x18\x03 \x01(\r:\x04\x32\x30\x30\x30\x12\x1f\n\x11min_after_dequeue\x18\x04 \x01(\r:\x04\x31\x30\x30\x30\x12\x15\n\nnum_epochs\x18\x05 \x01(\r:\x01\x30\x12\x17\n\x0bnum_readers\x18\x06 \x01(\r:\x02\x33\x32\x12\x1a\n\rprefetch_size\x18\r \x01(\r:\x03\x35\x31\x32\x12\"\n\x16num_parallel_map_calls\x18\x0e \x01(\r:\x02\x36\x34\x12\"\n\x13load_instance_masks\x18\x07 \x01(\x08:\x05\x66\x61lse\x12M\n\tmask_type\x18\n \x01(\x0e\x32).object_detection.protos.InstanceMaskType:\x0fNUMERICAL_MASKS\x12N\n\x16tf_record_input_reader\x18\x08 \x01(\x0b\x32,.object_detection.protos.TFRecordInputReaderH\x00\x12M\n\x15\x65xternal_input_reader\x18\t \x01(\x0b\x32,.object_detection.protos.ExternalInputReaderH\x00\x42\x0e\n\x0cinput_reader\")\n\x13TFRecordInputReader\x12\x12\n\ninput_path\x18\x01 \x03(\t\"\x1c\n\x13\x45xternalInputReader*\x05\x08\x01\x10\xe8\x07*C\n\x10InstanceMaskType\x12\x0b\n\x07\x44\x45\x46\x41ULT\x10\x00\x12\x13\n\x0fNUMERICAL_MASKS\x10\x01\x12\r\n\tPNG_MASKS\x10\x02')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'object_detection.protos.input_reader_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _INSTANCEMASKTYPE._serialized_start=753
  _INSTANCEMASKTYPE._serialized_end=820
  _INPUTREADER._serialized_start=72
  _INPUTREADER._serialized_end=678
  _TFRECORDINPUTREADER._serialized_start=680
  _TFRECORDINPUTREADER._serialized_end=721
  _EXTERNALINPUTREADER._serialized_start=723
  _EXTERNALINPUTREADER._serialized_end=751
# @@protoc_insertion_point(module_scope)