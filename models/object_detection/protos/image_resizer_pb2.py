# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: object_detection/protos/image_resizer.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n+object_detection/protos/image_resizer.proto\x12\x17object_detection.protos\"\xc6\x01\n\x0cImageResizer\x12T\n\x19keep_aspect_ratio_resizer\x18\x01 \x01(\x0b\x32/.object_detection.protos.KeepAspectRatioResizerH\x00\x12I\n\x13\x66ixed_shape_resizer\x18\x02 \x01(\x0b\x32*.object_detection.protos.FixedShapeResizerH\x00\x42\x15\n\x13image_resizer_oneof\"\xe1\x01\n\x16KeepAspectRatioResizer\x12\x1a\n\rmin_dimension\x18\x01 \x01(\x05:\x03\x36\x30\x30\x12\x1b\n\rmax_dimension\x18\x02 \x01(\x05:\x04\x31\x30\x32\x34\x12\x44\n\rresize_method\x18\x03 \x01(\x0e\x32#.object_detection.protos.ResizeType:\x08\x42ILINEAR\x12#\n\x14pad_to_max_dimension\x18\x04 \x01(\x08:\x05\x66\x61lse\x12#\n\x14\x63onvert_to_grayscale\x18\x05 \x01(\x08:\x05\x66\x61lse\"\xa7\x01\n\x11\x46ixedShapeResizer\x12\x13\n\x06height\x18\x01 \x01(\x05:\x03\x33\x30\x30\x12\x12\n\x05width\x18\x02 \x01(\x05:\x03\x33\x30\x30\x12\x44\n\rresize_method\x18\x03 \x01(\x0e\x32#.object_detection.protos.ResizeType:\x08\x42ILINEAR\x12#\n\x14\x63onvert_to_grayscale\x18\x04 \x01(\x08:\x05\x66\x61lse*G\n\nResizeType\x12\x0c\n\x08\x42ILINEAR\x10\x00\x12\x14\n\x10NEAREST_NEIGHBOR\x10\x01\x12\x0b\n\x07\x42ICUBIC\x10\x02\x12\x08\n\x04\x41REA\x10\x03')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'object_detection.protos.image_resizer_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _RESIZETYPE._serialized_start=671
  _RESIZETYPE._serialized_end=742
  _IMAGERESIZER._serialized_start=73
  _IMAGERESIZER._serialized_end=271
  _KEEPASPECTRATIORESIZER._serialized_start=274
  _KEEPASPECTRATIORESIZER._serialized_end=499
  _FIXEDSHAPERESIZER._serialized_start=502
  _FIXEDSHAPERESIZER._serialized_end=669
# @@protoc_insertion_point(module_scope)
