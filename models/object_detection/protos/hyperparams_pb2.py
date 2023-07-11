# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: object_detection/protos/hyperparams.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n)object_detection/protos/hyperparams.proto\x12\x17object_detection.protos\"\x87\x03\n\x0bHyperparams\x12\x39\n\x02op\x18\x01 \x01(\x0e\x32\'.object_detection.protos.Hyperparams.Op:\x04\x43ONV\x12\x39\n\x0bregularizer\x18\x02 \x01(\x0b\x32$.object_detection.protos.Regularizer\x12\x39\n\x0binitializer\x18\x03 \x01(\x0b\x32$.object_detection.protos.Initializer\x12I\n\nactivation\x18\x04 \x01(\x0e\x32/.object_detection.protos.Hyperparams.Activation:\x04RELU\x12\x36\n\nbatch_norm\x18\x05 \x01(\x0b\x32\".object_detection.protos.BatchNorm\"\x16\n\x02Op\x12\x08\n\x04\x43ONV\x10\x01\x12\x06\n\x02\x46\x43\x10\x02\",\n\nActivation\x12\x08\n\x04NONE\x10\x00\x12\x08\n\x04RELU\x10\x01\x12\n\n\x06RELU_6\x10\x02\"\xa6\x01\n\x0bRegularizer\x12@\n\x0el1_regularizer\x18\x01 \x01(\x0b\x32&.object_detection.protos.L1RegularizerH\x00\x12@\n\x0el2_regularizer\x18\x02 \x01(\x0b\x32&.object_detection.protos.L2RegularizerH\x00\x42\x13\n\x11regularizer_oneof\"\"\n\rL1Regularizer\x12\x11\n\x06weight\x18\x01 \x01(\x02:\x01\x31\"\"\n\rL2Regularizer\x12\x11\n\x06weight\x18\x01 \x01(\x02:\x01\x31\"\xb3\x02\n\x0bInitializer\x12[\n\x1ctruncated_normal_initializer\x18\x01 \x01(\x0b\x32\x33.object_detection.protos.TruncatedNormalInitializerH\x00\x12[\n\x1cvariance_scaling_initializer\x18\x02 \x01(\x0b\x32\x33.object_detection.protos.VarianceScalingInitializerH\x00\x12U\n\x19random_normal_initializer\x18\x03 \x01(\x0b\x32\x30.object_detection.protos.RandomNormalInitializerH\x00\x42\x13\n\x11initializer_oneof\"@\n\x1aTruncatedNormalInitializer\x12\x0f\n\x04mean\x18\x01 \x01(\x02:\x01\x30\x12\x11\n\x06stddev\x18\x02 \x01(\x02:\x01\x31\"\xc5\x01\n\x1aVarianceScalingInitializer\x12\x11\n\x06\x66\x61\x63tor\x18\x01 \x01(\x02:\x01\x32\x12\x16\n\x07uniform\x18\x02 \x01(\x08:\x05\x66\x61lse\x12N\n\x04mode\x18\x03 \x01(\x0e\x32\x38.object_detection.protos.VarianceScalingInitializer.Mode:\x06\x46\x41N_IN\",\n\x04Mode\x12\n\n\x06\x46\x41N_IN\x10\x00\x12\x0b\n\x07\x46\x41N_OUT\x10\x01\x12\x0b\n\x07\x46\x41N_AVG\x10\x02\"=\n\x17RandomNormalInitializer\x12\x0f\n\x04mean\x18\x01 \x01(\x02:\x01\x30\x12\x11\n\x06stddev\x18\x02 \x01(\x02:\x01\x31\"z\n\tBatchNorm\x12\x14\n\x05\x64\x65\x63\x61y\x18\x01 \x01(\x02:\x05\x30.999\x12\x14\n\x06\x63\x65nter\x18\x02 \x01(\x08:\x04true\x12\x14\n\x05scale\x18\x03 \x01(\x08:\x05\x66\x61lse\x12\x16\n\x07\x65psilon\x18\x04 \x01(\x02:\x05\x30.001\x12\x13\n\x05train\x18\x05 \x01(\x08:\x04true')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'object_detection.protos.hyperparams_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _HYPERPARAMS._serialized_start=71
  _HYPERPARAMS._serialized_end=462
  _HYPERPARAMS_OP._serialized_start=394
  _HYPERPARAMS_OP._serialized_end=416
  _HYPERPARAMS_ACTIVATION._serialized_start=418
  _HYPERPARAMS_ACTIVATION._serialized_end=462
  _REGULARIZER._serialized_start=465
  _REGULARIZER._serialized_end=631
  _L1REGULARIZER._serialized_start=633
  _L1REGULARIZER._serialized_end=667
  _L2REGULARIZER._serialized_start=669
  _L2REGULARIZER._serialized_end=703
  _INITIALIZER._serialized_start=706
  _INITIALIZER._serialized_end=1013
  _TRUNCATEDNORMALINITIALIZER._serialized_start=1015
  _TRUNCATEDNORMALINITIALIZER._serialized_end=1079
  _VARIANCESCALINGINITIALIZER._serialized_start=1082
  _VARIANCESCALINGINITIALIZER._serialized_end=1279
  _VARIANCESCALINGINITIALIZER_MODE._serialized_start=1235
  _VARIANCESCALINGINITIALIZER_MODE._serialized_end=1279
  _RANDOMNORMALINITIALIZER._serialized_start=1281
  _RANDOMNORMALINITIALIZER._serialized_end=1342
  _BATCHNORM._serialized_start=1344
  _BATCHNORM._serialized_end=1466
# @@protoc_insertion_point(module_scope)
