Ô·"
Î¤
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
¥
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.8.02v2.8.0-0-g3f878cff5b68¿!

embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¸dÈ*'
shared_nameembedding_1/embeddings

*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings* 
_output_shapes
:
¸dÈ*
dtype0

lstm_3/lstm_cell_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
È°	**
shared_namelstm_3/lstm_cell_3/kernel

-lstm_3/lstm_cell_3/kernel/Read/ReadVariableOpReadVariableOplstm_3/lstm_cell_3/kernel* 
_output_shapes
:
È°	*
dtype0
¤
#lstm_3/lstm_cell_3/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¬°	*4
shared_name%#lstm_3/lstm_cell_3/recurrent_kernel

7lstm_3/lstm_cell_3/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_3/lstm_cell_3/recurrent_kernel* 
_output_shapes
:
¬°	*
dtype0

lstm_3/lstm_cell_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:°	*(
shared_namelstm_3/lstm_cell_3/bias

+lstm_3/lstm_cell_3/bias/Read/ReadVariableOpReadVariableOplstm_3/lstm_cell_3/bias*
_output_shapes	
:°	*
dtype0

time_distributed/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¬¸d*(
shared_nametime_distributed/kernel

+time_distributed/kernel/Read/ReadVariableOpReadVariableOptime_distributed/kernel* 
_output_shapes
:
¬¸d*
dtype0

time_distributed/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:¸d*&
shared_nametime_distributed/bias
|
)time_distributed/bias/Read/ReadVariableOpReadVariableOptime_distributed/bias*
_output_shapes	
:¸d*
dtype0

NoOpNoOp
¸!
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ó 
valueé Bæ  Bß 
æ
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
 

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
* 
* 
Á
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses*
* 

	 layer
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses*
.
0
'1
(2
)3
*4
+5*
.
0
'1
(2
)3
*4
+5*
* 
°
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

1serving_default* 
jd
VARIABLE_VALUEembedding_1/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 

2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
ã
7
state_size

'kernel
(recurrent_kernel
)bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<_random_generator
=__call__
*>&call_and_return_all_conditional_losses*
* 

'0
(1
)2*

'0
(1
)2*
* 


?states
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
¦

*kernel
+bias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses*

*0
+1*

*0
+1*
* 

Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
* 
* 
YS
VARIABLE_VALUElstm_3/lstm_cell_3/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#lstm_3/lstm_cell_3/recurrent_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUElstm_3/lstm_cell_3/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEtime_distributed/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEtime_distributed/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
5
0
1
2
3
4
5
6*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

'0
(1
)2*

'0
(1
)2*
* 

Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
8	variables
9trainable_variables
:regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 

*0
+1*

*0
+1*
* 

Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*
* 
* 
* 

 0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

serving_default_input_2Placeholder*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*%
shape:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
|
serving_default_input_5Placeholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ¬
|
serving_default_input_6Placeholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ¬

serving_default_input_7Placeholder*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>¬*
dtype0*!
shape:ÿÿÿÿÿÿÿÿÿ>¬
è
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2serving_default_input_5serving_default_input_6serving_default_input_7embedding_1/embeddingslstm_3/lstm_cell_3/kernellstm_3/lstm_cell_3/bias#lstm_3/lstm_cell_3/recurrent_kerneltime_distributed/kerneltime_distributed/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d*(
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_100125
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
½
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*embedding_1/embeddings/Read/ReadVariableOp-lstm_3/lstm_cell_3/kernel/Read/ReadVariableOp7lstm_3/lstm_cell_3/recurrent_kernel/Read/ReadVariableOp+lstm_3/lstm_cell_3/bias/Read/ReadVariableOp+time_distributed/kernel/Read/ReadVariableOp)time_distributed/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_101810
À
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_1/embeddingslstm_3/lstm_cell_3/kernel#lstm_3/lstm_cell_3/recurrent_kernellstm_3/lstm_cell_3/biastime_distributed/kerneltime_distributed/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_101838ù¿ 
Ò

,__inference_embedding_1_layer_call_fn_100132

inputs
unknown:
¸dÈ
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_98565}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
Ç
__inference__traced_save_101810
file_prefix5
1savev2_embedding_1_embeddings_read_readvariableop8
4savev2_lstm_3_lstm_cell_3_kernel_read_readvariableopB
>savev2_lstm_3_lstm_cell_3_recurrent_kernel_read_readvariableop6
2savev2_lstm_3_lstm_cell_3_bias_read_readvariableop6
2savev2_time_distributed_kernel_read_readvariableop4
0savev2_time_distributed_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*·
value­BªB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH{
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B ù
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_1_embeddings_read_readvariableop4savev2_lstm_3_lstm_cell_3_kernel_read_readvariableop>savev2_lstm_3_lstm_cell_3_recurrent_kernel_read_readvariableop2savev2_lstm_3_lstm_cell_3_bias_read_readvariableop2savev2_time_distributed_kernel_read_readvariableop0savev2_time_distributed_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*U
_input_shapesD
B: :
¸dÈ:
È°	:
¬°	:°	:
¬¸d:¸d: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
¸dÈ:&"
 
_output_shapes
:
È°	:&"
 
_output_shapes
:
¬°	:!

_output_shapes	
:°	:&"
 
_output_shapes
:
¬¸d:!

_output_shapes	
:¸d:

_output_shapes
: 
§
Ø
'__inference_lstm_3_layer_call_fn_100172
inputs_0
unknown:
È°	
	unknown_0:	°	
	unknown_1:
¬°	
identity

identity_1

identity_2¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_98447}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
inputs/0

¡
1__inference_time_distributed_layer_call_fn_101438

inputs
unknown:
¬¸d
	unknown_0:	¸d
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_98535}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
Ë

B__inference_lstm_3_layer_call_and_return_conditional_losses_100824
inputs_0=
)lstm_cell_3_split_readvariableop_resource:
È°	:
+lstm_cell_3_split_1_readvariableop_resource:	°	7
#lstm_cell_3_readvariableop_resource:
¬°	
identity

identity_1

identity_2¢lstm_cell_3/ReadVariableOp¢lstm_cell_3/ReadVariableOp_1¢lstm_cell_3/ReadVariableOp_2¢lstm_cell_3/ReadVariableOp_3¢ lstm_cell_3/split/ReadVariableOp¢"lstm_cell_3/split_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¬s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¬w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_maskc
lstm_cell_3/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_3/ones_likeFill$lstm_cell_3/ones_like/Shape:output:0$lstm_cell_3/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ^
lstm_cell_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?
lstm_cell_3/dropout/MulMullstm_cell_3/ones_like:output:0"lstm_cell_3/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈg
lstm_cell_3/dropout/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:¥
0lstm_cell_3/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_3/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0g
"lstm_cell_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>Ë
 lstm_cell_3/dropout/GreaterEqualGreaterEqual9lstm_cell_3/dropout/random_uniform/RandomUniform:output:0+lstm_cell_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/dropout/CastCast$lstm_cell_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/dropout/Mul_1Mullstm_cell_3/dropout/Mul:z:0lstm_cell_3/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ`
lstm_cell_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?
lstm_cell_3/dropout_1/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈi
lstm_cell_3/dropout_1/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:©
2lstm_cell_3/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0i
$lstm_cell_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>Ñ
"lstm_cell_3/dropout_1/GreaterEqualGreaterEqual;lstm_cell_3/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/dropout_1/CastCast&lstm_cell_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/dropout_1/Mul_1Mullstm_cell_3/dropout_1/Mul:z:0lstm_cell_3/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ`
lstm_cell_3/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?
lstm_cell_3/dropout_2/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈi
lstm_cell_3/dropout_2/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:©
2lstm_cell_3/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0i
$lstm_cell_3/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>Ñ
"lstm_cell_3/dropout_2/GreaterEqualGreaterEqual;lstm_cell_3/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/dropout_2/CastCast&lstm_cell_3/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/dropout_2/Mul_1Mullstm_cell_3/dropout_2/Mul:z:0lstm_cell_3/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ`
lstm_cell_3/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?
lstm_cell_3/dropout_3/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈi
lstm_cell_3/dropout_3/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:©
2lstm_cell_3/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0i
$lstm_cell_3/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>Ñ
"lstm_cell_3/dropout_3/GreaterEqualGreaterEqual;lstm_cell_3/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/dropout_3/CastCast&lstm_cell_3/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/dropout_3/Mul_1Mullstm_cell_3/dropout_3/Mul:z:0lstm_cell_3/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ[
lstm_cell_3/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¢
lstm_cell_3/ones_like_1Fill&lstm_cell_3/ones_like_1/Shape:output:0&lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
lstm_cell_3/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_3/dropout_4/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_4/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
lstm_cell_3/dropout_4/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:©
2lstm_cell_3/dropout_4/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_4/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0i
$lstm_cell_3/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ñ
"lstm_cell_3/dropout_4/GreaterEqualGreaterEqual;lstm_cell_3/dropout_4/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/dropout_4/CastCast&lstm_cell_3/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/dropout_4/Mul_1Mullstm_cell_3/dropout_4/Mul:z:0lstm_cell_3/dropout_4/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
lstm_cell_3/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_3/dropout_5/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
lstm_cell_3/dropout_5/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:©
2lstm_cell_3/dropout_5/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_5/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0i
$lstm_cell_3/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ñ
"lstm_cell_3/dropout_5/GreaterEqualGreaterEqual;lstm_cell_3/dropout_5/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/dropout_5/CastCast&lstm_cell_3/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/dropout_5/Mul_1Mullstm_cell_3/dropout_5/Mul:z:0lstm_cell_3/dropout_5/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
lstm_cell_3/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_3/dropout_6/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_6/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
lstm_cell_3/dropout_6/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:©
2lstm_cell_3/dropout_6/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_6/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0i
$lstm_cell_3/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ñ
"lstm_cell_3/dropout_6/GreaterEqualGreaterEqual;lstm_cell_3/dropout_6/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/dropout_6/CastCast&lstm_cell_3/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/dropout_6/Mul_1Mullstm_cell_3/dropout_6/Mul:z:0lstm_cell_3/dropout_6/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
lstm_cell_3/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_3/dropout_7/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_7/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
lstm_cell_3/dropout_7/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:©
2lstm_cell_3/dropout_7/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_7/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0i
$lstm_cell_3/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ñ
"lstm_cell_3/dropout_7/GreaterEqualGreaterEqual;lstm_cell_3/dropout_7/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/dropout_7/CastCast&lstm_cell_3/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/dropout_7/Mul_1Mullstm_cell_3/dropout_7/Mul:z:0lstm_cell_3/dropout_7/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/mulMulstrided_slice_2:output:0lstm_cell_3/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/mul_1Mulstrided_slice_2:output:0lstm_cell_3/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/mul_2Mulstrided_slice_2:output:0lstm_cell_3/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/mul_3Mulstrided_slice_2:output:0lstm_cell_3/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ]
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_cell_3/split/ReadVariableOpReadVariableOp)lstm_cell_3_split_readvariableop_resource* 
_output_shapes
:
È°	*
dtype0Ê
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0(lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È¬:
È¬:
È¬:
È¬*
	num_split
lstm_cell_3/MatMulMatMullstm_cell_3/mul:z:0lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/MatMul_1MatMullstm_cell_3/mul_1:z:0lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/MatMul_2MatMullstm_cell_3/mul_2:z:0lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/MatMul_3MatMullstm_cell_3/mul_3:z:0lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬_
lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"lstm_cell_3/split_1/ReadVariableOpReadVariableOp+lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:°	*
dtype0¼
lstm_cell_3/split_1Split&lstm_cell_3/split_1/split_dim:output:0*lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:¬:¬:¬:¬*
	num_split
lstm_cell_3/BiasAddBiasAddlstm_cell_3/MatMul:product:0lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/BiasAdd_1BiasAddlstm_cell_3/MatMul_1:product:0lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/BiasAdd_2BiasAddlstm_cell_3/MatMul_2:product:0lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/BiasAdd_3BiasAddlstm_cell_3/MatMul_3:product:0lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬|
lstm_cell_3/mul_4Mulzeros:output:0lstm_cell_3/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬|
lstm_cell_3/mul_5Mulzeros:output:0lstm_cell_3/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬|
lstm_cell_3/mul_6Mulzeros:output:0lstm_cell_3/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬|
lstm_cell_3/mul_7Mulzeros:output:0lstm_cell_3/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/ReadVariableOpReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0p
lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  r
!lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell_3/strided_sliceStridedSlice"lstm_cell_3/ReadVariableOp:value:0(lstm_cell_3/strided_slice/stack:output:0*lstm_cell_3/strided_slice/stack_1:output:0*lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
lstm_cell_3/MatMul_4MatMullstm_cell_3/mul_4:z:0"lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/addAddV2lstm_cell_3/BiasAdd:output:0lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
lstm_cell_3/SigmoidSigmoidlstm_cell_3/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/ReadVariableOp_1ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0r
!lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  t
#lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  t
#lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
lstm_cell_3/strided_slice_1StridedSlice$lstm_cell_3/ReadVariableOp_1:value:0*lstm_cell_3/strided_slice_1/stack:output:0,lstm_cell_3/strided_slice_1/stack_1:output:0,lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
lstm_cell_3/MatMul_5MatMullstm_cell_3/mul_5:z:0$lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/add_1AddV2lstm_cell_3/BiasAdd_1:output:0lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬x
lstm_cell_3/mul_8Mullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/ReadVariableOp_2ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0r
!lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  t
#lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
lstm_cell_3/strided_slice_2StridedSlice$lstm_cell_3/ReadVariableOp_2:value:0*lstm_cell_3/strided_slice_2/stack:output:0,lstm_cell_3/strided_slice_2/stack_1:output:0,lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
lstm_cell_3/MatMul_6MatMullstm_cell_3/mul_6:z:0$lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/add_2AddV2lstm_cell_3/BiasAdd_2:output:0lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬b
lstm_cell_3/TanhTanhlstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬z
lstm_cell_3/mul_9Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬{
lstm_cell_3/add_3AddV2lstm_cell_3/mul_8:z:0lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/ReadVariableOp_3ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0r
!lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
lstm_cell_3/strided_slice_3StridedSlice$lstm_cell_3/ReadVariableOp_3:value:0*lstm_cell_3/strided_slice_3/stack:output:0,lstm_cell_3/strided_slice_3/stack_1:output:0,lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
lstm_cell_3/MatMul_7MatMullstm_cell_3/mul_7:z:0$lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/add_4AddV2lstm_cell_3/BiasAdd_3:output:0lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬d
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/mul_10Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ù
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_3_split_readvariableop_resource+lstm_cell_3_split_1_readvariableop_resource#lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_100624*
condR
while_cond_100623*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
NoOpNoOp^lstm_cell_3/ReadVariableOp^lstm_cell_3/ReadVariableOp_1^lstm_cell_3/ReadVariableOp_2^lstm_cell_3/ReadVariableOp_3!^lstm_cell_3/split/ReadVariableOp#^lstm_cell_3/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ: : : 28
lstm_cell_3/ReadVariableOplstm_cell_3/ReadVariableOp2<
lstm_cell_3/ReadVariableOp_1lstm_cell_3/ReadVariableOp_12<
lstm_cell_3/ReadVariableOp_2lstm_cell_3/ReadVariableOp_22<
lstm_cell_3/ReadVariableOp_3lstm_cell_3/ReadVariableOp_32D
 lstm_cell_3/split/ReadVariableOp lstm_cell_3/split/ReadVariableOp2H
"lstm_cell_3/split_1/ReadVariableOp"lstm_cell_3/split_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
inputs/0
´
¾
while_cond_98064
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_98064___redundant_placeholder03
/while_while_cond_98064___redundant_placeholder13
/while_while_cond_98064___redundant_placeholder23
/while_while_cond_98064___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
:
·

L__inference_time_distributed_layer_call_and_return_conditional_losses_101460

inputs8
$dense_matmul_readvariableop_resource:
¬¸d4
%dense_biasadd_readvariableop_resource:	¸d
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
¬¸d*
dtype0
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸d
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:¸d*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸dc
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸d\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :¸d
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapedense/Softmax:softmax:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸do
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
Æ
ø
B__inference_model_3_layer_call_and_return_conditional_losses_98822

inputs
inputs_1
inputs_2
inputs_3%
embedding_1_98566:
¸dÈ 
lstm_3_98803:
È°	
lstm_3_98805:	°	 
lstm_3_98807:
¬°	*
time_distributed_98812:
¬¸d%
time_distributed_98814:	¸d
identity

identity_1

identity_2¢#embedding_1/StatefulPartitionedCall¢lstm_3/StatefulPartitionedCall¢(time_distributed/StatefulPartitionedCallõ
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1_98566*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_98565ì
lstm_3/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0inputs_2inputs_3lstm_3_98803lstm_3_98805lstm_3_98807*
Tin

2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_98802¿
(time_distributed/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0time_distributed_98812time_distributed_98814*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_98496o
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  ¨
time_distributed/ReshapeReshape'lstm_3/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
IdentityIdentity1time_distributed/StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸dy

Identity_1Identity'lstm_3/StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬y

Identity_2Identity'lstm_3/StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¸
NoOpNoOp$^embedding_1/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ>¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : : 2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:TP
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>¬
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs

Ç
K__inference_time_distributed_layer_call_and_return_conditional_losses_98535

inputs
dense_98525:
¬¸d
dense_98527:	¸d
identity¢dense/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ï
dense/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_98525dense_98527*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_98485\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :¸d
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape&dense/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸do
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸df
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
÷Â
	
while_body_100624
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
1while_lstm_cell_3_split_readvariableop_resource_0:
È°	B
3while_lstm_cell_3_split_1_readvariableop_resource_0:	°	?
+while_lstm_cell_3_readvariableop_resource_0:
¬°	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
/while_lstm_cell_3_split_readvariableop_resource:
È°	@
1while_lstm_cell_3_split_1_readvariableop_resource:	°	=
)while_lstm_cell_3_readvariableop_resource:
¬°	¢ while/lstm_cell_3/ReadVariableOp¢"while/lstm_cell_3/ReadVariableOp_1¢"while/lstm_cell_3/ReadVariableOp_2¢"while/lstm_cell_3/ReadVariableOp_3¢&while/lstm_cell_3/split/ReadVariableOp¢(while/lstm_cell_3/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype0
!while/lstm_cell_3/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?®
while/lstm_cell_3/ones_likeFill*while/lstm_cell_3/ones_like/Shape:output:0*while/lstm_cell_3/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈd
while/lstm_cell_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?§
while/lstm_cell_3/dropout/MulMul$while/lstm_cell_3/ones_like:output:0(while/lstm_cell_3/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈs
while/lstm_cell_3/dropout/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:±
6while/lstm_cell_3/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_3/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0m
(while/lstm_cell_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>Ý
&while/lstm_cell_3/dropout/GreaterEqualGreaterEqual?while/lstm_cell_3/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
while/lstm_cell_3/dropout/CastCast*while/lstm_cell_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
while/lstm_cell_3/dropout/Mul_1Mul!while/lstm_cell_3/dropout/Mul:z:0"while/lstm_cell_3/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈf
!while/lstm_cell_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?«
while/lstm_cell_3/dropout_1/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈu
!while/lstm_cell_3/dropout_1/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:µ
8while/lstm_cell_3/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0o
*while/lstm_cell_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>ã
(while/lstm_cell_3/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 while/lstm_cell_3/dropout_1/CastCast,while/lstm_cell_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¦
!while/lstm_cell_3/dropout_1/Mul_1Mul#while/lstm_cell_3/dropout_1/Mul:z:0$while/lstm_cell_3/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈf
!while/lstm_cell_3/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?«
while/lstm_cell_3/dropout_2/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈu
!while/lstm_cell_3/dropout_2/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:µ
8while/lstm_cell_3/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0o
*while/lstm_cell_3/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>ã
(while/lstm_cell_3/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 while/lstm_cell_3/dropout_2/CastCast,while/lstm_cell_3/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¦
!while/lstm_cell_3/dropout_2/Mul_1Mul#while/lstm_cell_3/dropout_2/Mul:z:0$while/lstm_cell_3/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈf
!while/lstm_cell_3/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?«
while/lstm_cell_3/dropout_3/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈu
!while/lstm_cell_3/dropout_3/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:µ
8while/lstm_cell_3/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0o
*while/lstm_cell_3/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>ã
(while/lstm_cell_3/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 while/lstm_cell_3/dropout_3/CastCast,while/lstm_cell_3/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¦
!while/lstm_cell_3/dropout_3/Mul_1Mul#while/lstm_cell_3/dropout_3/Mul:z:0$while/lstm_cell_3/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈf
#while/lstm_cell_3/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?´
while/lstm_cell_3/ones_like_1Fill,while/lstm_cell_3/ones_like_1/Shape:output:0,while/lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
!while/lstm_cell_3/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?­
while/lstm_cell_3/dropout_4/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_4/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬w
!while/lstm_cell_3/dropout_4/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:µ
8while/lstm_cell_3/dropout_4/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_4/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0o
*while/lstm_cell_3/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>ã
(while/lstm_cell_3/dropout_4/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_4/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 while/lstm_cell_3/dropout_4/CastCast,while/lstm_cell_3/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
!while/lstm_cell_3/dropout_4/Mul_1Mul#while/lstm_cell_3/dropout_4/Mul:z:0$while/lstm_cell_3/dropout_4/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
!while/lstm_cell_3/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?­
while/lstm_cell_3/dropout_5/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬w
!while/lstm_cell_3/dropout_5/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:µ
8while/lstm_cell_3/dropout_5/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_5/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0o
*while/lstm_cell_3/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>ã
(while/lstm_cell_3/dropout_5/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_5/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 while/lstm_cell_3/dropout_5/CastCast,while/lstm_cell_3/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
!while/lstm_cell_3/dropout_5/Mul_1Mul#while/lstm_cell_3/dropout_5/Mul:z:0$while/lstm_cell_3/dropout_5/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
!while/lstm_cell_3/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?­
while/lstm_cell_3/dropout_6/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_6/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬w
!while/lstm_cell_3/dropout_6/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:µ
8while/lstm_cell_3/dropout_6/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_6/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0o
*while/lstm_cell_3/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>ã
(while/lstm_cell_3/dropout_6/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_6/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 while/lstm_cell_3/dropout_6/CastCast,while/lstm_cell_3/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
!while/lstm_cell_3/dropout_6/Mul_1Mul#while/lstm_cell_3/dropout_6/Mul:z:0$while/lstm_cell_3/dropout_6/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
!while/lstm_cell_3/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?­
while/lstm_cell_3/dropout_7/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_7/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬w
!while/lstm_cell_3/dropout_7/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:µ
8while/lstm_cell_3/dropout_7/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_7/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0o
*while/lstm_cell_3/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>ã
(while/lstm_cell_3/dropout_7/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_7/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 while/lstm_cell_3/dropout_7/CastCast,while/lstm_cell_3/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
!while/lstm_cell_3/dropout_7/Mul_1Mul#while/lstm_cell_3/dropout_7/Mul:z:0$while/lstm_cell_3/dropout_7/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
while/lstm_cell_3/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell_3/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈª
while/lstm_cell_3/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈª
while/lstm_cell_3/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈª
while/lstm_cell_3/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈc
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&while/lstm_cell_3/split/ReadVariableOpReadVariableOp1while_lstm_cell_3_split_readvariableop_resource_0* 
_output_shapes
:
È°	*
dtype0Ü
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0.while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È¬:
È¬:
È¬:
È¬*
	num_split
while/lstm_cell_3/MatMulMatMulwhile/lstm_cell_3/mul:z:0 while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/MatMul_1MatMulwhile/lstm_cell_3/mul_1:z:0 while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/MatMul_2MatMulwhile/lstm_cell_3/mul_2:z:0 while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/MatMul_3MatMulwhile/lstm_cell_3/mul_3:z:0 while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
#while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:°	*
dtype0Î
while/lstm_cell_3/split_1Split,while/lstm_cell_3/split_1/split_dim:output:00while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:¬:¬:¬:¬*
	num_split
while/lstm_cell_3/BiasAddBiasAdd"while/lstm_cell_3/MatMul:product:0"while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/lstm_cell_3/BiasAdd_1BiasAdd$while/lstm_cell_3/MatMul_1:product:0"while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/lstm_cell_3/BiasAdd_2BiasAdd$while/lstm_cell_3/MatMul_2:product:0"while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/lstm_cell_3/BiasAdd_3BiasAdd$while/lstm_cell_3/MatMul_3:product:0"while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_4Mulwhile_placeholder_2%while/lstm_cell_3/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_5Mulwhile_placeholder_2%while/lstm_cell_3/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_6Mulwhile_placeholder_2%while/lstm_cell_3/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_7Mulwhile_placeholder_2%while/lstm_cell_3/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 while/lstm_cell_3/ReadVariableOpReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0v
%while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  x
'while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell_3/strided_sliceStridedSlice(while/lstm_cell_3/ReadVariableOp:value:0.while/lstm_cell_3/strided_slice/stack:output:00while/lstm_cell_3/strided_slice/stack_1:output:00while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_4MatMulwhile/lstm_cell_3/mul_4:z:0(while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/addAddV2"while/lstm_cell_3/BiasAdd:output:0$while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r
while/lstm_cell_3/SigmoidSigmoidwhile/lstm_cell_3/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"while/lstm_cell_3/ReadVariableOp_1ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0x
'while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  z
)while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  z
)while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ñ
!while/lstm_cell_3/strided_slice_1StridedSlice*while/lstm_cell_3/ReadVariableOp_1:value:00while/lstm_cell_3/strided_slice_1/stack:output:02while/lstm_cell_3/strided_slice_1/stack_1:output:02while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask 
while/lstm_cell_3/MatMul_5MatMulwhile/lstm_cell_3/mul_5:z:0*while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/add_1AddV2$while/lstm_cell_3/BiasAdd_1:output:0$while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬v
while/lstm_cell_3/Sigmoid_1Sigmoidwhile/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_8Mulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"while/lstm_cell_3/ReadVariableOp_2ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0x
'while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  z
)while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ñ
!while/lstm_cell_3/strided_slice_2StridedSlice*while/lstm_cell_3/ReadVariableOp_2:value:00while/lstm_cell_3/strided_slice_2/stack:output:02while/lstm_cell_3/strided_slice_2/stack_1:output:02while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask 
while/lstm_cell_3/MatMul_6MatMulwhile/lstm_cell_3/mul_6:z:0*while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/add_2AddV2$while/lstm_cell_3/BiasAdd_2:output:0$while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬n
while/lstm_cell_3/TanhTanhwhile/lstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_9Mulwhile/lstm_cell_3/Sigmoid:y:0while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/add_3AddV2while/lstm_cell_3/mul_8:z:0while/lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"while/lstm_cell_3/ReadVariableOp_3ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0x
'while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ñ
!while/lstm_cell_3/strided_slice_3StridedSlice*while/lstm_cell_3/ReadVariableOp_3:value:00while/lstm_cell_3/strided_slice_3/stack:output:02while/lstm_cell_3/strided_slice_3/stack_1:output:02while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask 
while/lstm_cell_3/MatMul_7MatMulwhile/lstm_cell_3/mul_7:z:0*while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/add_4AddV2$while/lstm_cell_3/BiasAdd_3:output:0$while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬v
while/lstm_cell_3/Sigmoid_2Sigmoidwhile/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬p
while/lstm_cell_3/Tanh_1Tanhwhile/lstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_10Mulwhile/lstm_cell_3/Sigmoid_2:y:0while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_3/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬y
while/Identity_5Identitywhile/lstm_cell_3/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬²

while/NoOpNoOp!^while/lstm_cell_3/ReadVariableOp#^while/lstm_cell_3/ReadVariableOp_1#^while/lstm_cell_3/ReadVariableOp_2#^while/lstm_cell_3/ReadVariableOp_3'^while/lstm_cell_3/split/ReadVariableOp)^while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_3_readvariableop_resource+while_lstm_cell_3_readvariableop_resource_0"h
1while_lstm_cell_3_split_1_readvariableop_resource3while_lstm_cell_3_split_1_readvariableop_resource_0"d
/while_lstm_cell_3_split_readvariableop_resource1while_lstm_cell_3_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : 2D
 while/lstm_cell_3/ReadVariableOp while/lstm_cell_3/ReadVariableOp2H
"while/lstm_cell_3/ReadVariableOp_1"while/lstm_cell_3/ReadVariableOp_12H
"while/lstm_cell_3/ReadVariableOp_2"while/lstm_cell_3/ReadVariableOp_22H
"while/lstm_cell_3/ReadVariableOp_3"while/lstm_cell_3/ReadVariableOp_32P
&while/lstm_cell_3/split/ReadVariableOp&while/lstm_cell_3/split/ReadVariableOp2T
(while/lstm_cell_3/split_1/ReadVariableOp(while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
: 
ÿt
	
while_body_100922
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
1while_lstm_cell_3_split_readvariableop_resource_0:
È°	B
3while_lstm_cell_3_split_1_readvariableop_resource_0:	°	?
+while_lstm_cell_3_readvariableop_resource_0:
¬°	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
/while_lstm_cell_3_split_readvariableop_resource:
È°	@
1while_lstm_cell_3_split_1_readvariableop_resource:	°	=
)while_lstm_cell_3_readvariableop_resource:
¬°	¢ while/lstm_cell_3/ReadVariableOp¢"while/lstm_cell_3/ReadVariableOp_1¢"while/lstm_cell_3/ReadVariableOp_2¢"while/lstm_cell_3/ReadVariableOp_3¢&while/lstm_cell_3/split/ReadVariableOp¢(while/lstm_cell_3/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype0
!while/lstm_cell_3/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?®
while/lstm_cell_3/ones_likeFill*while/lstm_cell_3/ones_like/Shape:output:0*while/lstm_cell_3/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈf
#while/lstm_cell_3/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?´
while/lstm_cell_3/ones_like_1Fill,while/lstm_cell_3/ones_like_1/Shape:output:0,while/lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬§
while/lstm_cell_3/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ©
while/lstm_cell_3/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ©
while/lstm_cell_3/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ©
while/lstm_cell_3/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈc
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&while/lstm_cell_3/split/ReadVariableOpReadVariableOp1while_lstm_cell_3_split_readvariableop_resource_0* 
_output_shapes
:
È°	*
dtype0Ü
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0.while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È¬:
È¬:
È¬:
È¬*
	num_split
while/lstm_cell_3/MatMulMatMulwhile/lstm_cell_3/mul:z:0 while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/MatMul_1MatMulwhile/lstm_cell_3/mul_1:z:0 while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/MatMul_2MatMulwhile/lstm_cell_3/mul_2:z:0 while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/MatMul_3MatMulwhile/lstm_cell_3/mul_3:z:0 while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
#while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:°	*
dtype0Î
while/lstm_cell_3/split_1Split,while/lstm_cell_3/split_1/split_dim:output:00while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:¬:¬:¬:¬*
	num_split
while/lstm_cell_3/BiasAddBiasAdd"while/lstm_cell_3/MatMul:product:0"while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/lstm_cell_3/BiasAdd_1BiasAdd$while/lstm_cell_3/MatMul_1:product:0"while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/lstm_cell_3/BiasAdd_2BiasAdd$while/lstm_cell_3/MatMul_2:product:0"while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/lstm_cell_3/BiasAdd_3BiasAdd$while/lstm_cell_3/MatMul_3:product:0"while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_4Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_5Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_6Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_7Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 while/lstm_cell_3/ReadVariableOpReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0v
%while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  x
'while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell_3/strided_sliceStridedSlice(while/lstm_cell_3/ReadVariableOp:value:0.while/lstm_cell_3/strided_slice/stack:output:00while/lstm_cell_3/strided_slice/stack_1:output:00while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_4MatMulwhile/lstm_cell_3/mul_4:z:0(while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/addAddV2"while/lstm_cell_3/BiasAdd:output:0$while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r
while/lstm_cell_3/SigmoidSigmoidwhile/lstm_cell_3/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"while/lstm_cell_3/ReadVariableOp_1ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0x
'while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  z
)while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  z
)while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ñ
!while/lstm_cell_3/strided_slice_1StridedSlice*while/lstm_cell_3/ReadVariableOp_1:value:00while/lstm_cell_3/strided_slice_1/stack:output:02while/lstm_cell_3/strided_slice_1/stack_1:output:02while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask 
while/lstm_cell_3/MatMul_5MatMulwhile/lstm_cell_3/mul_5:z:0*while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/add_1AddV2$while/lstm_cell_3/BiasAdd_1:output:0$while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬v
while/lstm_cell_3/Sigmoid_1Sigmoidwhile/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_8Mulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"while/lstm_cell_3/ReadVariableOp_2ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0x
'while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  z
)while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ñ
!while/lstm_cell_3/strided_slice_2StridedSlice*while/lstm_cell_3/ReadVariableOp_2:value:00while/lstm_cell_3/strided_slice_2/stack:output:02while/lstm_cell_3/strided_slice_2/stack_1:output:02while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask 
while/lstm_cell_3/MatMul_6MatMulwhile/lstm_cell_3/mul_6:z:0*while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/add_2AddV2$while/lstm_cell_3/BiasAdd_2:output:0$while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬n
while/lstm_cell_3/TanhTanhwhile/lstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_9Mulwhile/lstm_cell_3/Sigmoid:y:0while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/add_3AddV2while/lstm_cell_3/mul_8:z:0while/lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"while/lstm_cell_3/ReadVariableOp_3ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0x
'while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ñ
!while/lstm_cell_3/strided_slice_3StridedSlice*while/lstm_cell_3/ReadVariableOp_3:value:00while/lstm_cell_3/strided_slice_3/stack:output:02while/lstm_cell_3/strided_slice_3/stack_1:output:02while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask 
while/lstm_cell_3/MatMul_7MatMulwhile/lstm_cell_3/mul_7:z:0*while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/add_4AddV2$while/lstm_cell_3/BiasAdd_3:output:0$while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬v
while/lstm_cell_3/Sigmoid_2Sigmoidwhile/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬p
while/lstm_cell_3/Tanh_1Tanhwhile/lstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_10Mulwhile/lstm_cell_3/Sigmoid_2:y:0while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_3/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬y
while/Identity_5Identitywhile/lstm_cell_3/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬²

while/NoOpNoOp!^while/lstm_cell_3/ReadVariableOp#^while/lstm_cell_3/ReadVariableOp_1#^while/lstm_cell_3/ReadVariableOp_2#^while/lstm_cell_3/ReadVariableOp_3'^while/lstm_cell_3/split/ReadVariableOp)^while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_3_readvariableop_resource+while_lstm_cell_3_readvariableop_resource_0"h
1while_lstm_cell_3_split_1_readvariableop_resource3while_lstm_cell_3_split_1_readvariableop_resource_0"d
/while_lstm_cell_3_split_readvariableop_resource1while_lstm_cell_3_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : 2D
 while/lstm_cell_3/ReadVariableOp while/lstm_cell_3/ReadVariableOp2H
"while/lstm_cell_3/ReadVariableOp_1"while/lstm_cell_3/ReadVariableOp_12H
"while/lstm_cell_3/ReadVariableOp_2"while/lstm_cell_3/ReadVariableOp_22H
"while/lstm_cell_3/ReadVariableOp_3"while/lstm_cell_3/ReadVariableOp_32P
&while/lstm_cell_3/split/ReadVariableOp&while/lstm_cell_3/split/ReadVariableOp2T
(while/lstm_cell_3/split_1/ReadVariableOp(while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
: 
éz
©
B__inference_lstm_3_layer_call_and_return_conditional_losses_101058

inputs
initial_state_0
initial_state_1=
)lstm_cell_3_split_readvariableop_resource:
È°	:
+lstm_cell_3_split_1_readvariableop_resource:	°	7
#lstm_cell_3_readvariableop_resource:
¬°	
identity

identity_1

identity_2¢lstm_cell_3/ReadVariableOp¢lstm_cell_3/ReadVariableOp_1¢lstm_cell_3/ReadVariableOp_2¢lstm_cell_3/ReadVariableOp_3¢ lstm_cell_3/split/ReadVariableOp¢"lstm_cell_3/split_1/ReadVariableOp¢whilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈB
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_maskc
lstm_cell_3/ones_like/ShapeShapestrided_slice_1:output:0*
T0*
_output_shapes
:`
lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_3/ones_likeFill$lstm_cell_3/ones_like/Shape:output:0$lstm_cell_3/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ\
lstm_cell_3/ones_like_1/ShapeShapeinitial_state_0*
T0*
_output_shapes
:b
lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¢
lstm_cell_3/ones_like_1Fill&lstm_cell_3/ones_like_1/Shape:output:0&lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/mulMulstrided_slice_1:output:0lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/mul_1Mulstrided_slice_1:output:0lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/mul_2Mulstrided_slice_1:output:0lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/mul_3Mulstrided_slice_1:output:0lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ]
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_cell_3/split/ReadVariableOpReadVariableOp)lstm_cell_3_split_readvariableop_resource* 
_output_shapes
:
È°	*
dtype0Ê
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0(lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È¬:
È¬:
È¬:
È¬*
	num_split
lstm_cell_3/MatMulMatMullstm_cell_3/mul:z:0lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/MatMul_1MatMullstm_cell_3/mul_1:z:0lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/MatMul_2MatMullstm_cell_3/mul_2:z:0lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/MatMul_3MatMullstm_cell_3/mul_3:z:0lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬_
lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"lstm_cell_3/split_1/ReadVariableOpReadVariableOp+lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:°	*
dtype0¼
lstm_cell_3/split_1Split&lstm_cell_3/split_1/split_dim:output:0*lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:¬:¬:¬:¬*
	num_split
lstm_cell_3/BiasAddBiasAddlstm_cell_3/MatMul:product:0lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/BiasAdd_1BiasAddlstm_cell_3/MatMul_1:product:0lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/BiasAdd_2BiasAddlstm_cell_3/MatMul_2:product:0lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/BiasAdd_3BiasAddlstm_cell_3/MatMul_3:product:0lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬~
lstm_cell_3/mul_4Mulinitial_state_0 lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬~
lstm_cell_3/mul_5Mulinitial_state_0 lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬~
lstm_cell_3/mul_6Mulinitial_state_0 lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬~
lstm_cell_3/mul_7Mulinitial_state_0 lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/ReadVariableOpReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0p
lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  r
!lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell_3/strided_sliceStridedSlice"lstm_cell_3/ReadVariableOp:value:0(lstm_cell_3/strided_slice/stack:output:0*lstm_cell_3/strided_slice/stack_1:output:0*lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
lstm_cell_3/MatMul_4MatMullstm_cell_3/mul_4:z:0"lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/addAddV2lstm_cell_3/BiasAdd:output:0lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
lstm_cell_3/SigmoidSigmoidlstm_cell_3/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/ReadVariableOp_1ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0r
!lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  t
#lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  t
#lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
lstm_cell_3/strided_slice_1StridedSlice$lstm_cell_3/ReadVariableOp_1:value:0*lstm_cell_3/strided_slice_1/stack:output:0,lstm_cell_3/strided_slice_1/stack_1:output:0,lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
lstm_cell_3/MatMul_5MatMullstm_cell_3/mul_5:z:0$lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/add_1AddV2lstm_cell_3/BiasAdd_1:output:0lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬w
lstm_cell_3/mul_8Mullstm_cell_3/Sigmoid_1:y:0initial_state_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/ReadVariableOp_2ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0r
!lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  t
#lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
lstm_cell_3/strided_slice_2StridedSlice$lstm_cell_3/ReadVariableOp_2:value:0*lstm_cell_3/strided_slice_2/stack:output:0,lstm_cell_3/strided_slice_2/stack_1:output:0,lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
lstm_cell_3/MatMul_6MatMullstm_cell_3/mul_6:z:0$lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/add_2AddV2lstm_cell_3/BiasAdd_2:output:0lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬b
lstm_cell_3/TanhTanhlstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬z
lstm_cell_3/mul_9Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬{
lstm_cell_3/add_3AddV2lstm_cell_3/mul_8:z:0lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/ReadVariableOp_3ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0r
!lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
lstm_cell_3/strided_slice_3StridedSlice$lstm_cell_3/ReadVariableOp_3:value:0*lstm_cell_3/strided_slice_3/stack:output:0,lstm_cell_3/strided_slice_3/stack_1:output:0,lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
lstm_cell_3/MatMul_7MatMullstm_cell_3/mul_7:z:0$lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/add_4AddV2lstm_cell_3/BiasAdd_3:output:0lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬d
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/mul_10Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÷
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0initial_state_0initial_state_1strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_3_split_readvariableop_resource+lstm_cell_3_split_1_readvariableop_resource#lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_100922*
condR
while_cond_100921*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
NoOpNoOp^lstm_cell_3/ReadVariableOp^lstm_cell_3/ReadVariableOp_1^lstm_cell_3/ReadVariableOp_2^lstm_cell_3/ReadVariableOp_3!^lstm_cell_3/split/ReadVariableOp#^lstm_cell_3/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : 28
lstm_cell_3/ReadVariableOplstm_cell_3/ReadVariableOp2<
lstm_cell_3/ReadVariableOp_1lstm_cell_3/ReadVariableOp_12<
lstm_cell_3/ReadVariableOp_2lstm_cell_3/ReadVariableOp_22<
lstm_cell_3/ReadVariableOp_3lstm_cell_3/ReadVariableOp_32D
 lstm_cell_3/split/ReadVariableOp lstm_cell_3/split/ReadVariableOp2H
"lstm_cell_3/split_1/ReadVariableOp"lstm_cell_3/split_1/ReadVariableOp2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:YU
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
)
_user_specified_nameinitial_state/0:YU
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
)
_user_specified_nameinitial_state/1
§
Ø
'__inference_lstm_3_layer_call_fn_100157
inputs_0
unknown:
È°	
	unknown_0:	°	
	unknown_1:
¬°	
identity

identity_1

identity_2¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_98136}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
inputs/0
á"
Ù
while_body_98376
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_3_98400_0:
È°	(
while_lstm_cell_3_98402_0:	°	-
while_lstm_cell_3_98404_0:
¬°	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_3_98400:
È°	&
while_lstm_cell_3_98402:	°	+
while_lstm_cell_3_98404:
¬°	¢)while/lstm_cell_3/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype0°
)while/lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_3_98400_0while_lstm_cell_3_98402_0while_lstm_cell_3_98404_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_98317Û
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_3/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity2while/lstm_cell_3/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/Identity_5Identity2while/lstm_cell_3/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬x

while/NoOpNoOp*^while/lstm_cell_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_3_98400while_lstm_cell_3_98400_0"4
while_lstm_cell_3_98402while_lstm_cell_3_98402_0"4
while_lstm_cell_3_98404while_lstm_cell_3_98404_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : 2V
)while/lstm_cell_3/StatefulPartitionedCall)while/lstm_cell_3/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
: 
®
¬
G__inference_lstm_cell_3_layer_call_and_return_conditional_losses_101744

inputs
states_0
states_11
split_readvariableop_resource:
È°	.
split_1_readvariableop_resource:	°	+
readvariableop_resource:
¬°	
identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?q
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈO
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈT
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?u
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈQ
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>­
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈt
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈp
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈT
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?u
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈQ
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>­
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈt
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈp
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈT
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?u
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈQ
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>­
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈt
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈp
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈI
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬T
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬S
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0]
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>­
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬t
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬p
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬T
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬S
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0]
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>­
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬t
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬p
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬T
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬S
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0]
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>­
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬t
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬p
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬T
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬S
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0]
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>­
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬t
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬p
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬X
mulMulinputsdropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ\
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ\
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ\
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
È°	*
dtype0¦
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È¬:
È¬:
È¬:
È¬*
	num_split\
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:°	*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:¬:¬:¬:¬*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬^
mul_4Mulstates_0dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬^
mul_5Mulstates_0dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬^
mul_6Mulstates_0dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬^
mul_7Mulstates_0dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
¬°	*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskh
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
¬°	*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskj
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬X
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
¬°	*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskj
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬V
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬W
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
¬°	*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskj
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬R
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬L
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬[
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
IdentityIdentity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬\

Identity_1Identity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬[

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬À
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
states/1
Õ
è
model_3_lstm_3_while_cond_97777:
6model_3_lstm_3_while_model_3_lstm_3_while_loop_counter@
<model_3_lstm_3_while_model_3_lstm_3_while_maximum_iterations$
 model_3_lstm_3_while_placeholder&
"model_3_lstm_3_while_placeholder_1&
"model_3_lstm_3_while_placeholder_2&
"model_3_lstm_3_while_placeholder_3:
6model_3_lstm_3_while_less_model_3_lstm_3_strided_sliceQ
Mmodel_3_lstm_3_while_model_3_lstm_3_while_cond_97777___redundant_placeholder0Q
Mmodel_3_lstm_3_while_model_3_lstm_3_while_cond_97777___redundant_placeholder1Q
Mmodel_3_lstm_3_while_model_3_lstm_3_while_cond_97777___redundant_placeholder2Q
Mmodel_3_lstm_3_while_model_3_lstm_3_while_cond_97777___redundant_placeholder3!
model_3_lstm_3_while_identity

model_3/lstm_3/while/LessLess model_3_lstm_3_while_placeholder6model_3_lstm_3_while_less_model_3_lstm_3_strided_slice*
T0*
_output_shapes
: i
model_3/lstm_3/while/IdentityIdentitymodel_3/lstm_3/while/Less:z:0*
T0
*
_output_shapes
: "G
model_3_lstm_3_while_identity&model_3/lstm_3/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
:
Í

'__inference_lstm_3_layer_call_fn_100189

inputs
initial_state_0
initial_state_1
unknown:
È°	
	unknown_0:	°	
	unknown_1:
¬°	
identity

identity_1

identity_2¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputsinitial_state_0initial_state_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_98802}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:YU
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
)
_user_specified_nameinitial_state/0:YU
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
)
_user_specified_nameinitial_state/1
ù
÷
,__inference_lstm_cell_3_layer_call_fn_101499

inputs
states_0
states_1
unknown:
È°	
	unknown_0:	°	
	unknown_1:
¬°	
identity

identity_1

identity_2¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_98051p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
states/1
´
¾
while_cond_98375
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_98375___redundant_placeholder03
/while_while_cond_98375___redundant_placeholder13
/while_while_cond_98375___redundant_placeholder23
/while_while_cond_98375___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
:
¹½

 __inference__wrapped_model_97934
input_2
input_7
input_5
input_6>
*model_3_embedding_1_embedding_lookup_97680:
¸dÈL
8model_3_lstm_3_lstm_cell_3_split_readvariableop_resource:
È°	I
:model_3_lstm_3_lstm_cell_3_split_1_readvariableop_resource:	°	F
2model_3_lstm_3_lstm_cell_3_readvariableop_resource:
¬°	Q
=model_3_time_distributed_dense_matmul_readvariableop_resource:
¬¸dM
>model_3_time_distributed_dense_biasadd_readvariableop_resource:	¸d
identity

identity_1

identity_2¢$model_3/embedding_1/embedding_lookup¢)model_3/lstm_3/lstm_cell_3/ReadVariableOp¢+model_3/lstm_3/lstm_cell_3/ReadVariableOp_1¢+model_3/lstm_3/lstm_cell_3/ReadVariableOp_2¢+model_3/lstm_3/lstm_cell_3/ReadVariableOp_3¢/model_3/lstm_3/lstm_cell_3/split/ReadVariableOp¢1model_3/lstm_3/lstm_cell_3/split_1/ReadVariableOp¢model_3/lstm_3/while¢5model_3/time_distributed/dense/BiasAdd/ReadVariableOp¢4model_3/time_distributed/dense/MatMul/ReadVariableOps
model_3/embedding_1/CastCastinput_2*

DstT0*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
$model_3/embedding_1/embedding_lookupResourceGather*model_3_embedding_1_embedding_lookup_97680model_3/embedding_1/Cast:y:0*
Tindices0*=
_class3
1/loc:@model_3/embedding_1/embedding_lookup/97680*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*
dtype0ç
-model_3/embedding_1/embedding_lookup/IdentityIdentity-model_3/embedding_1/embedding_lookup:output:0*
T0*=
_class3
1/loc:@model_3/embedding_1/embedding_lookup/97680*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ³
/model_3/embedding_1/embedding_lookup/Identity_1Identity6model_3/embedding_1/embedding_lookup/Identity:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈr
model_3/lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ç
model_3/lstm_3/transpose	Transpose8model_3/embedding_1/embedding_lookup/Identity_1:output:0&model_3/lstm_3/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ`
model_3/lstm_3/ShapeShapemodel_3/lstm_3/transpose:y:0*
T0*
_output_shapes
:l
"model_3/lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$model_3/lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$model_3/lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
model_3/lstm_3/strided_sliceStridedSlicemodel_3/lstm_3/Shape:output:0+model_3/lstm_3/strided_slice/stack:output:0-model_3/lstm_3/strided_slice/stack_1:output:0-model_3/lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
*model_3/lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿß
model_3/lstm_3/TensorArrayV2TensorListReserve3model_3/lstm_3/TensorArrayV2/element_shape:output:0%model_3/lstm_3/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Dmodel_3/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   
6model_3/lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_3/lstm_3/transpose:y:0Mmodel_3/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
$model_3/lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&model_3/lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&model_3/lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
model_3/lstm_3/strided_slice_1StridedSlicemodel_3/lstm_3/transpose:y:0-model_3/lstm_3/strided_slice_1/stack:output:0/model_3/lstm_3/strided_slice_1/stack_1:output:0/model_3/lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask
*model_3/lstm_3/lstm_cell_3/ones_like/ShapeShape'model_3/lstm_3/strided_slice_1:output:0*
T0*
_output_shapes
:o
*model_3/lstm_3/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?É
$model_3/lstm_3/lstm_cell_3/ones_likeFill3model_3/lstm_3/lstm_cell_3/ones_like/Shape:output:03model_3/lstm_3/lstm_cell_3/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈc
,model_3/lstm_3/lstm_cell_3/ones_like_1/ShapeShapeinput_5*
T0*
_output_shapes
:q
,model_3/lstm_3/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ï
&model_3/lstm_3/lstm_cell_3/ones_like_1Fill5model_3/lstm_3/lstm_cell_3/ones_like_1/Shape:output:05model_3/lstm_3/lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬°
model_3/lstm_3/lstm_cell_3/mulMul'model_3/lstm_3/strided_slice_1:output:0-model_3/lstm_3/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ²
 model_3/lstm_3/lstm_cell_3/mul_1Mul'model_3/lstm_3/strided_slice_1:output:0-model_3/lstm_3/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ²
 model_3/lstm_3/lstm_cell_3/mul_2Mul'model_3/lstm_3/strided_slice_1:output:0-model_3/lstm_3/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ²
 model_3/lstm_3/lstm_cell_3/mul_3Mul'model_3/lstm_3/strided_slice_1:output:0-model_3/lstm_3/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈl
*model_3/lstm_3/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ª
/model_3/lstm_3/lstm_cell_3/split/ReadVariableOpReadVariableOp8model_3_lstm_3_lstm_cell_3_split_readvariableop_resource* 
_output_shapes
:
È°	*
dtype0÷
 model_3/lstm_3/lstm_cell_3/splitSplit3model_3/lstm_3/lstm_cell_3/split/split_dim:output:07model_3/lstm_3/lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È¬:
È¬:
È¬:
È¬*
	num_split­
!model_3/lstm_3/lstm_cell_3/MatMulMatMul"model_3/lstm_3/lstm_cell_3/mul:z:0)model_3/lstm_3/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬±
#model_3/lstm_3/lstm_cell_3/MatMul_1MatMul$model_3/lstm_3/lstm_cell_3/mul_1:z:0)model_3/lstm_3/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬±
#model_3/lstm_3/lstm_cell_3/MatMul_2MatMul$model_3/lstm_3/lstm_cell_3/mul_2:z:0)model_3/lstm_3/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬±
#model_3/lstm_3/lstm_cell_3/MatMul_3MatMul$model_3/lstm_3/lstm_cell_3/mul_3:z:0)model_3/lstm_3/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬n
,model_3/lstm_3/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ©
1model_3/lstm_3/lstm_cell_3/split_1/ReadVariableOpReadVariableOp:model_3_lstm_3_lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:°	*
dtype0é
"model_3/lstm_3/lstm_cell_3/split_1Split5model_3/lstm_3/lstm_cell_3/split_1/split_dim:output:09model_3/lstm_3/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:¬:¬:¬:¬*
	num_splitº
"model_3/lstm_3/lstm_cell_3/BiasAddBiasAdd+model_3/lstm_3/lstm_cell_3/MatMul:product:0+model_3/lstm_3/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¾
$model_3/lstm_3/lstm_cell_3/BiasAdd_1BiasAdd-model_3/lstm_3/lstm_cell_3/MatMul_1:product:0+model_3/lstm_3/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¾
$model_3/lstm_3/lstm_cell_3/BiasAdd_2BiasAdd-model_3/lstm_3/lstm_cell_3/MatMul_2:product:0+model_3/lstm_3/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¾
$model_3/lstm_3/lstm_cell_3/BiasAdd_3BiasAdd-model_3/lstm_3/lstm_cell_3/MatMul_3:product:0+model_3/lstm_3/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 model_3/lstm_3/lstm_cell_3/mul_4Mulinput_5/model_3/lstm_3/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 model_3/lstm_3/lstm_cell_3/mul_5Mulinput_5/model_3/lstm_3/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 model_3/lstm_3/lstm_cell_3/mul_6Mulinput_5/model_3/lstm_3/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 model_3/lstm_3/lstm_cell_3/mul_7Mulinput_5/model_3/lstm_3/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
)model_3/lstm_3/lstm_cell_3/ReadVariableOpReadVariableOp2model_3_lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0
.model_3/lstm_3/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
0model_3/lstm_3/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  
0model_3/lstm_3/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
(model_3/lstm_3/lstm_cell_3/strided_sliceStridedSlice1model_3/lstm_3/lstm_cell_3/ReadVariableOp:value:07model_3/lstm_3/lstm_cell_3/strided_slice/stack:output:09model_3/lstm_3/lstm_cell_3/strided_slice/stack_1:output:09model_3/lstm_3/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask¹
#model_3/lstm_3/lstm_cell_3/MatMul_4MatMul$model_3/lstm_3/lstm_cell_3/mul_4:z:01model_3/lstm_3/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¶
model_3/lstm_3/lstm_cell_3/addAddV2+model_3/lstm_3/lstm_cell_3/BiasAdd:output:0-model_3/lstm_3/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"model_3/lstm_3/lstm_cell_3/SigmoidSigmoid"model_3/lstm_3/lstm_cell_3/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
+model_3/lstm_3/lstm_cell_3/ReadVariableOp_1ReadVariableOp2model_3_lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0
0model_3/lstm_3/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  
2model_3/lstm_3/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  
2model_3/lstm_3/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      þ
*model_3/lstm_3/lstm_cell_3/strided_slice_1StridedSlice3model_3/lstm_3/lstm_cell_3/ReadVariableOp_1:value:09model_3/lstm_3/lstm_cell_3/strided_slice_1/stack:output:0;model_3/lstm_3/lstm_cell_3/strided_slice_1/stack_1:output:0;model_3/lstm_3/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask»
#model_3/lstm_3/lstm_cell_3/MatMul_5MatMul$model_3/lstm_3/lstm_cell_3/mul_5:z:03model_3/lstm_3/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬º
 model_3/lstm_3/lstm_cell_3/add_1AddV2-model_3/lstm_3/lstm_cell_3/BiasAdd_1:output:0-model_3/lstm_3/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
$model_3/lstm_3/lstm_cell_3/Sigmoid_1Sigmoid$model_3/lstm_3/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 model_3/lstm_3/lstm_cell_3/mul_8Mul(model_3/lstm_3/lstm_cell_3/Sigmoid_1:y:0input_6*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
+model_3/lstm_3/lstm_cell_3/ReadVariableOp_2ReadVariableOp2model_3_lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0
0model_3/lstm_3/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  
2model_3/lstm_3/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
2model_3/lstm_3/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      þ
*model_3/lstm_3/lstm_cell_3/strided_slice_2StridedSlice3model_3/lstm_3/lstm_cell_3/ReadVariableOp_2:value:09model_3/lstm_3/lstm_cell_3/strided_slice_2/stack:output:0;model_3/lstm_3/lstm_cell_3/strided_slice_2/stack_1:output:0;model_3/lstm_3/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask»
#model_3/lstm_3/lstm_cell_3/MatMul_6MatMul$model_3/lstm_3/lstm_cell_3/mul_6:z:03model_3/lstm_3/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬º
 model_3/lstm_3/lstm_cell_3/add_2AddV2-model_3/lstm_3/lstm_cell_3/BiasAdd_2:output:0-model_3/lstm_3/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
model_3/lstm_3/lstm_cell_3/TanhTanh$model_3/lstm_3/lstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬§
 model_3/lstm_3/lstm_cell_3/mul_9Mul&model_3/lstm_3/lstm_cell_3/Sigmoid:y:0#model_3/lstm_3/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¨
 model_3/lstm_3/lstm_cell_3/add_3AddV2$model_3/lstm_3/lstm_cell_3/mul_8:z:0$model_3/lstm_3/lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
+model_3/lstm_3/lstm_cell_3/ReadVariableOp_3ReadVariableOp2model_3_lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0
0model_3/lstm_3/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
2model_3/lstm_3/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
2model_3/lstm_3/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      þ
*model_3/lstm_3/lstm_cell_3/strided_slice_3StridedSlice3model_3/lstm_3/lstm_cell_3/ReadVariableOp_3:value:09model_3/lstm_3/lstm_cell_3/strided_slice_3/stack:output:0;model_3/lstm_3/lstm_cell_3/strided_slice_3/stack_1:output:0;model_3/lstm_3/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask»
#model_3/lstm_3/lstm_cell_3/MatMul_7MatMul$model_3/lstm_3/lstm_cell_3/mul_7:z:03model_3/lstm_3/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬º
 model_3/lstm_3/lstm_cell_3/add_4AddV2-model_3/lstm_3/lstm_cell_3/BiasAdd_3:output:0-model_3/lstm_3/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
$model_3/lstm_3/lstm_cell_3/Sigmoid_2Sigmoid$model_3/lstm_3/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
!model_3/lstm_3/lstm_cell_3/Tanh_1Tanh$model_3/lstm_3/lstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
!model_3/lstm_3/lstm_cell_3/mul_10Mul(model_3/lstm_3/lstm_cell_3/Sigmoid_2:y:0%model_3/lstm_3/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬}
,model_3/lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  ã
model_3/lstm_3/TensorArrayV2_1TensorListReserve5model_3/lstm_3/TensorArrayV2_1/element_shape:output:0%model_3/lstm_3/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒU
model_3/lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : r
'model_3/lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿc
!model_3/lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
model_3/lstm_3/whileWhile*model_3/lstm_3/while/loop_counter:output:00model_3/lstm_3/while/maximum_iterations:output:0model_3/lstm_3/time:output:0'model_3/lstm_3/TensorArrayV2_1:handle:0input_5input_6%model_3/lstm_3/strided_slice:output:0Fmodel_3/lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:08model_3_lstm_3_lstm_cell_3_split_readvariableop_resource:model_3_lstm_3_lstm_cell_3_split_1_readvariableop_resource2model_3_lstm_3_lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *+
body#R!
model_3_lstm_3_while_body_97778*+
cond#R!
model_3_lstm_3_while_cond_97777*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *
parallel_iterations 
?model_3/lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  ù
1model_3/lstm_3/TensorArrayV2Stack/TensorListStackTensorListStackmodel_3/lstm_3/while:output:3Hmodel_3/lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬*
element_dtype0w
$model_3/lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿp
&model_3/lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&model_3/lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
model_3/lstm_3/strided_slice_2StridedSlice:model_3/lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0-model_3/lstm_3/strided_slice_2/stack:output:0/model_3/lstm_3/strided_slice_2/stack_1:output:0/model_3/lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_maskt
model_3/lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Í
model_3/lstm_3/transpose_1	Transpose:model_3/lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0(model_3/lstm_3/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬j
model_3/lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
model_3/time_distributed/ShapeShapemodel_3/lstm_3/transpose_1:y:0*
T0*
_output_shapes
:v
,model_3/time_distributed/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.model_3/time_distributed/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model_3/time_distributed/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&model_3/time_distributed/strided_sliceStridedSlice'model_3/time_distributed/Shape:output:05model_3/time_distributed/strided_slice/stack:output:07model_3/time_distributed/strided_slice/stack_1:output:07model_3/time_distributed/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
&model_3/time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  ¯
 model_3/time_distributed/ReshapeReshapemodel_3/lstm_3/transpose_1:y:0/model_3/time_distributed/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬´
4model_3/time_distributed/dense/MatMul/ReadVariableOpReadVariableOp=model_3_time_distributed_dense_matmul_readvariableop_resource* 
_output_shapes
:
¬¸d*
dtype0Ë
%model_3/time_distributed/dense/MatMulMatMul)model_3/time_distributed/Reshape:output:0<model_3/time_distributed/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸d±
5model_3/time_distributed/dense/BiasAdd/ReadVariableOpReadVariableOp>model_3_time_distributed_dense_biasadd_readvariableop_resource*
_output_shapes	
:¸d*
dtype0Ô
&model_3/time_distributed/dense/BiasAddBiasAdd/model_3/time_distributed/dense/MatMul:product:0=model_3/time_distributed/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸d
&model_3/time_distributed/dense/SoftmaxSoftmax/model_3/time_distributed/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸du
*model_3/time_distributed/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿm
*model_3/time_distributed/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :¸dù
(model_3/time_distributed/Reshape_1/shapePack3model_3/time_distributed/Reshape_1/shape/0:output:0/model_3/time_distributed/strided_slice:output:03model_3/time_distributed/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:Ò
"model_3/time_distributed/Reshape_1Reshape0model_3/time_distributed/dense/Softmax:softmax:01model_3/time_distributed/Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸dy
(model_3/time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  ³
"model_3/time_distributed/Reshape_2Reshapemodel_3/lstm_3/transpose_1:y:01model_3/time_distributed/Reshape_2/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
IdentityIdentitymodel_3/lstm_3/while:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬o

Identity_1Identitymodel_3/lstm_3/while:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬

Identity_2Identity+model_3/time_distributed/Reshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d
NoOpNoOp%^model_3/embedding_1/embedding_lookup*^model_3/lstm_3/lstm_cell_3/ReadVariableOp,^model_3/lstm_3/lstm_cell_3/ReadVariableOp_1,^model_3/lstm_3/lstm_cell_3/ReadVariableOp_2,^model_3/lstm_3/lstm_cell_3/ReadVariableOp_30^model_3/lstm_3/lstm_cell_3/split/ReadVariableOp2^model_3/lstm_3/lstm_cell_3/split_1/ReadVariableOp^model_3/lstm_3/while6^model_3/time_distributed/dense/BiasAdd/ReadVariableOp5^model_3/time_distributed/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ>¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : : 2L
$model_3/embedding_1/embedding_lookup$model_3/embedding_1/embedding_lookup2V
)model_3/lstm_3/lstm_cell_3/ReadVariableOp)model_3/lstm_3/lstm_cell_3/ReadVariableOp2Z
+model_3/lstm_3/lstm_cell_3/ReadVariableOp_1+model_3/lstm_3/lstm_cell_3/ReadVariableOp_12Z
+model_3/lstm_3/lstm_cell_3/ReadVariableOp_2+model_3/lstm_3/lstm_cell_3/ReadVariableOp_22Z
+model_3/lstm_3/lstm_cell_3/ReadVariableOp_3+model_3/lstm_3/lstm_cell_3/ReadVariableOp_32b
/model_3/lstm_3/lstm_cell_3/split/ReadVariableOp/model_3/lstm_3/lstm_cell_3/split/ReadVariableOp2f
1model_3/lstm_3/lstm_cell_3/split_1/ReadVariableOp1model_3/lstm_3/lstm_cell_3/split_1/ReadVariableOp2,
model_3/lstm_3/whilemodel_3/lstm_3/while2n
5model_3/time_distributed/dense/BiasAdd/ReadVariableOp5model_3/time_distributed/dense/BiasAdd/ReadVariableOp2l
4model_3/time_distributed/dense/MatMul/ReadVariableOp4model_3/time_distributed/dense/MatMul/ReadVariableOp:Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:UQ
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>¬
!
_user_specified_name	input_7:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
!
_user_specified_name	input_5:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
!
_user_specified_name	input_6
¢
Ö
'__inference_model_3_layer_call_fn_98841
input_2
input_7
input_5
input_6
unknown:
¸dÈ
	unknown_0:
È°	
	unknown_1:	°	
	unknown_2:
¬°	
	unknown_3:
¬¸d
	unknown_4:	¸d
identity

identity_1

identity_2¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinput_2input_7input_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2
*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*(
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_98822}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸dr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ>¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:UQ
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>¬
!
_user_specified_name	input_7:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
!
_user_specified_name	input_5:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
!
_user_specified_name	input_6
®
Ú
'__inference_model_3_layer_call_fn_99425
inputs_0
inputs_1
inputs_2
inputs_3
unknown:
¸dÈ
	unknown_0:
È°	
	unknown_1:	°	
	unknown_2:
¬°	
	unknown_3:
¬¸d
	unknown_4:	¸d
identity

identity_1

identity_2¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2
*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*(
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_98822}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸dr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ>¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>¬
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
inputs/2:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
inputs/3
Ôz
¦
A__inference_lstm_3_layer_call_and_return_conditional_losses_98802

inputs
initial_state
initial_state_1=
)lstm_cell_3_split_readvariableop_resource:
È°	:
+lstm_cell_3_split_1_readvariableop_resource:	°	7
#lstm_cell_3_readvariableop_resource:
¬°	
identity

identity_1

identity_2¢lstm_cell_3/ReadVariableOp¢lstm_cell_3/ReadVariableOp_1¢lstm_cell_3/ReadVariableOp_2¢lstm_cell_3/ReadVariableOp_3¢ lstm_cell_3/split/ReadVariableOp¢"lstm_cell_3/split_1/ReadVariableOp¢whilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈB
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_maskc
lstm_cell_3/ones_like/ShapeShapestrided_slice_1:output:0*
T0*
_output_shapes
:`
lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_3/ones_likeFill$lstm_cell_3/ones_like/Shape:output:0$lstm_cell_3/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈZ
lstm_cell_3/ones_like_1/ShapeShapeinitial_state*
T0*
_output_shapes
:b
lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¢
lstm_cell_3/ones_like_1Fill&lstm_cell_3/ones_like_1/Shape:output:0&lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/mulMulstrided_slice_1:output:0lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/mul_1Mulstrided_slice_1:output:0lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/mul_2Mulstrided_slice_1:output:0lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/mul_3Mulstrided_slice_1:output:0lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ]
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_cell_3/split/ReadVariableOpReadVariableOp)lstm_cell_3_split_readvariableop_resource* 
_output_shapes
:
È°	*
dtype0Ê
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0(lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È¬:
È¬:
È¬:
È¬*
	num_split
lstm_cell_3/MatMulMatMullstm_cell_3/mul:z:0lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/MatMul_1MatMullstm_cell_3/mul_1:z:0lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/MatMul_2MatMullstm_cell_3/mul_2:z:0lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/MatMul_3MatMullstm_cell_3/mul_3:z:0lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬_
lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"lstm_cell_3/split_1/ReadVariableOpReadVariableOp+lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:°	*
dtype0¼
lstm_cell_3/split_1Split&lstm_cell_3/split_1/split_dim:output:0*lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:¬:¬:¬:¬*
	num_split
lstm_cell_3/BiasAddBiasAddlstm_cell_3/MatMul:product:0lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/BiasAdd_1BiasAddlstm_cell_3/MatMul_1:product:0lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/BiasAdd_2BiasAddlstm_cell_3/MatMul_2:product:0lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/BiasAdd_3BiasAddlstm_cell_3/MatMul_3:product:0lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬|
lstm_cell_3/mul_4Mulinitial_state lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬|
lstm_cell_3/mul_5Mulinitial_state lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬|
lstm_cell_3/mul_6Mulinitial_state lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬|
lstm_cell_3/mul_7Mulinitial_state lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/ReadVariableOpReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0p
lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  r
!lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell_3/strided_sliceStridedSlice"lstm_cell_3/ReadVariableOp:value:0(lstm_cell_3/strided_slice/stack:output:0*lstm_cell_3/strided_slice/stack_1:output:0*lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
lstm_cell_3/MatMul_4MatMullstm_cell_3/mul_4:z:0"lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/addAddV2lstm_cell_3/BiasAdd:output:0lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
lstm_cell_3/SigmoidSigmoidlstm_cell_3/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/ReadVariableOp_1ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0r
!lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  t
#lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  t
#lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
lstm_cell_3/strided_slice_1StridedSlice$lstm_cell_3/ReadVariableOp_1:value:0*lstm_cell_3/strided_slice_1/stack:output:0,lstm_cell_3/strided_slice_1/stack_1:output:0,lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
lstm_cell_3/MatMul_5MatMullstm_cell_3/mul_5:z:0$lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/add_1AddV2lstm_cell_3/BiasAdd_1:output:0lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬w
lstm_cell_3/mul_8Mullstm_cell_3/Sigmoid_1:y:0initial_state_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/ReadVariableOp_2ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0r
!lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  t
#lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
lstm_cell_3/strided_slice_2StridedSlice$lstm_cell_3/ReadVariableOp_2:value:0*lstm_cell_3/strided_slice_2/stack:output:0,lstm_cell_3/strided_slice_2/stack_1:output:0,lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
lstm_cell_3/MatMul_6MatMullstm_cell_3/mul_6:z:0$lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/add_2AddV2lstm_cell_3/BiasAdd_2:output:0lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬b
lstm_cell_3/TanhTanhlstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬z
lstm_cell_3/mul_9Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬{
lstm_cell_3/add_3AddV2lstm_cell_3/mul_8:z:0lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/ReadVariableOp_3ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0r
!lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
lstm_cell_3/strided_slice_3StridedSlice$lstm_cell_3/ReadVariableOp_3:value:0*lstm_cell_3/strided_slice_3/stack:output:0,lstm_cell_3/strided_slice_3/stack_1:output:0,lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
lstm_cell_3/MatMul_7MatMullstm_cell_3/mul_7:z:0$lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/add_4AddV2lstm_cell_3/BiasAdd_3:output:0lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬d
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/mul_10Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0initial_stateinitial_state_1strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_3_split_readvariableop_resource+lstm_cell_3_split_1_readvariableop_resource#lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_98666*
condR
while_cond_98665*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
NoOpNoOp^lstm_cell_3/ReadVariableOp^lstm_cell_3/ReadVariableOp_1^lstm_cell_3/ReadVariableOp_2^lstm_cell_3/ReadVariableOp_3!^lstm_cell_3/split/ReadVariableOp#^lstm_cell_3/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : 28
lstm_cell_3/ReadVariableOplstm_cell_3/ReadVariableOp2<
lstm_cell_3/ReadVariableOp_1lstm_cell_3/ReadVariableOp_12<
lstm_cell_3/ReadVariableOp_2lstm_cell_3/ReadVariableOp_22<
lstm_cell_3/ReadVariableOp_3lstm_cell_3/ReadVariableOp_32D
 lstm_cell_3/split/ReadVariableOp lstm_cell_3/split/ReadVariableOp2H
"lstm_cell_3/split_1/ReadVariableOp"lstm_cell_3/split_1/ReadVariableOp2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:WS
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
'
_user_specified_nameinitial_state:WS
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
'
_user_specified_nameinitial_state
æD
¬
G__inference_lstm_cell_3_layer_call_and_return_conditional_losses_101598

inputs
states_0
states_11
split_readvariableop_resource:
È°	.
split_1_readvariableop_resource:	°	+
readvariableop_resource:
¬°	
identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈI
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Y
mulMulinputsones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ[
mul_1Mulinputsones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ[
mul_2Mulinputsones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ[
mul_3Mulinputsones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
È°	*
dtype0¦
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È¬:
È¬:
È¬:
È¬*
	num_split\
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:°	*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:¬:¬:¬:¬*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬_
mul_4Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬_
mul_5Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬_
mul_6Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬_
mul_7Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
¬°	*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskh
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
¬°	*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskj
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬X
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
¬°	*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskj
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬V
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬W
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
¬°	*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskj
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬R
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬L
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬[
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
IdentityIdentity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬\

Identity_1Identity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬[

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬À
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
states/1
·:

A__inference_lstm_3_layer_call_and_return_conditional_losses_98136

inputs%
lstm_cell_3_98052:
È°	 
lstm_cell_3_98054:	°	%
lstm_cell_3_98056:
¬°	
identity

identity_1

identity_2¢#lstm_cell_3/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¬s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¬w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_maskò
#lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_3_98052lstm_cell_3_98054lstm_cell_3_98056*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_98051n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ³
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_3_98052lstm_cell_3_98054lstm_cell_3_98056*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_98065*
condR
while_cond_98064*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬t
NoOpNoOp$^lstm_cell_3/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ: : : 2J
#lstm_cell_3/StatefulPartitionedCall#lstm_cell_3/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
Å
¤
model_3_lstm_3_while_body_97778:
6model_3_lstm_3_while_model_3_lstm_3_while_loop_counter@
<model_3_lstm_3_while_model_3_lstm_3_while_maximum_iterations$
 model_3_lstm_3_while_placeholder&
"model_3_lstm_3_while_placeholder_1&
"model_3_lstm_3_while_placeholder_2&
"model_3_lstm_3_while_placeholder_37
3model_3_lstm_3_while_model_3_lstm_3_strided_slice_0u
qmodel_3_lstm_3_while_tensorarrayv2read_tensorlistgetitem_model_3_lstm_3_tensorarrayunstack_tensorlistfromtensor_0T
@model_3_lstm_3_while_lstm_cell_3_split_readvariableop_resource_0:
È°	Q
Bmodel_3_lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0:	°	N
:model_3_lstm_3_while_lstm_cell_3_readvariableop_resource_0:
¬°	!
model_3_lstm_3_while_identity#
model_3_lstm_3_while_identity_1#
model_3_lstm_3_while_identity_2#
model_3_lstm_3_while_identity_3#
model_3_lstm_3_while_identity_4#
model_3_lstm_3_while_identity_55
1model_3_lstm_3_while_model_3_lstm_3_strided_slices
omodel_3_lstm_3_while_tensorarrayv2read_tensorlistgetitem_model_3_lstm_3_tensorarrayunstack_tensorlistfromtensorR
>model_3_lstm_3_while_lstm_cell_3_split_readvariableop_resource:
È°	O
@model_3_lstm_3_while_lstm_cell_3_split_1_readvariableop_resource:	°	L
8model_3_lstm_3_while_lstm_cell_3_readvariableop_resource:
¬°	¢/model_3/lstm_3/while/lstm_cell_3/ReadVariableOp¢1model_3/lstm_3/while/lstm_cell_3/ReadVariableOp_1¢1model_3/lstm_3/while/lstm_cell_3/ReadVariableOp_2¢1model_3/lstm_3/while/lstm_cell_3/ReadVariableOp_3¢5model_3/lstm_3/while/lstm_cell_3/split/ReadVariableOp¢7model_3/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp
Fmodel_3/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   ò
8model_3/lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqmodel_3_lstm_3_while_tensorarrayv2read_tensorlistgetitem_model_3_lstm_3_tensorarrayunstack_tensorlistfromtensor_0 model_3_lstm_3_while_placeholderOmodel_3/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype0
0model_3/lstm_3/while/lstm_cell_3/ones_like/ShapeShape?model_3/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:u
0model_3/lstm_3/while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Û
*model_3/lstm_3/while/lstm_cell_3/ones_likeFill9model_3/lstm_3/while/lstm_cell_3/ones_like/Shape:output:09model_3/lstm_3/while/lstm_cell_3/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
2model_3/lstm_3/while/lstm_cell_3/ones_like_1/ShapeShape"model_3_lstm_3_while_placeholder_2*
T0*
_output_shapes
:w
2model_3/lstm_3/while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?á
,model_3/lstm_3/while/lstm_cell_3/ones_like_1Fill;model_3/lstm_3/while/lstm_cell_3/ones_like_1/Shape:output:0;model_3/lstm_3/while/lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ô
$model_3/lstm_3/while/lstm_cell_3/mulMul?model_3/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:03model_3/lstm_3/while/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÖ
&model_3/lstm_3/while/lstm_cell_3/mul_1Mul?model_3/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:03model_3/lstm_3/while/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÖ
&model_3/lstm_3/while/lstm_cell_3/mul_2Mul?model_3/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:03model_3/lstm_3/while/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÖ
&model_3/lstm_3/while/lstm_cell_3/mul_3Mul?model_3/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:03model_3/lstm_3/while/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈr
0model_3/lstm_3/while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¸
5model_3/lstm_3/while/lstm_cell_3/split/ReadVariableOpReadVariableOp@model_3_lstm_3_while_lstm_cell_3_split_readvariableop_resource_0* 
_output_shapes
:
È°	*
dtype0
&model_3/lstm_3/while/lstm_cell_3/splitSplit9model_3/lstm_3/while/lstm_cell_3/split/split_dim:output:0=model_3/lstm_3/while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È¬:
È¬:
È¬:
È¬*
	num_split¿
'model_3/lstm_3/while/lstm_cell_3/MatMulMatMul(model_3/lstm_3/while/lstm_cell_3/mul:z:0/model_3/lstm_3/while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ã
)model_3/lstm_3/while/lstm_cell_3/MatMul_1MatMul*model_3/lstm_3/while/lstm_cell_3/mul_1:z:0/model_3/lstm_3/while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ã
)model_3/lstm_3/while/lstm_cell_3/MatMul_2MatMul*model_3/lstm_3/while/lstm_cell_3/mul_2:z:0/model_3/lstm_3/while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ã
)model_3/lstm_3/while/lstm_cell_3/MatMul_3MatMul*model_3/lstm_3/while/lstm_cell_3/mul_3:z:0/model_3/lstm_3/while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬t
2model_3/lstm_3/while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ·
7model_3/lstm_3/while/lstm_cell_3/split_1/ReadVariableOpReadVariableOpBmodel_3_lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:°	*
dtype0û
(model_3/lstm_3/while/lstm_cell_3/split_1Split;model_3/lstm_3/while/lstm_cell_3/split_1/split_dim:output:0?model_3/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:¬:¬:¬:¬*
	num_splitÌ
(model_3/lstm_3/while/lstm_cell_3/BiasAddBiasAdd1model_3/lstm_3/while/lstm_cell_3/MatMul:product:01model_3/lstm_3/while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ð
*model_3/lstm_3/while/lstm_cell_3/BiasAdd_1BiasAdd3model_3/lstm_3/while/lstm_cell_3/MatMul_1:product:01model_3/lstm_3/while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ð
*model_3/lstm_3/while/lstm_cell_3/BiasAdd_2BiasAdd3model_3/lstm_3/while/lstm_cell_3/MatMul_2:product:01model_3/lstm_3/while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ð
*model_3/lstm_3/while/lstm_cell_3/BiasAdd_3BiasAdd3model_3/lstm_3/while/lstm_cell_3/MatMul_3:product:01model_3/lstm_3/while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬»
&model_3/lstm_3/while/lstm_cell_3/mul_4Mul"model_3_lstm_3_while_placeholder_25model_3/lstm_3/while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬»
&model_3/lstm_3/while/lstm_cell_3/mul_5Mul"model_3_lstm_3_while_placeholder_25model_3/lstm_3/while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬»
&model_3/lstm_3/while/lstm_cell_3/mul_6Mul"model_3_lstm_3_while_placeholder_25model_3/lstm_3/while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬»
&model_3/lstm_3/while/lstm_cell_3/mul_7Mul"model_3_lstm_3_while_placeholder_25model_3/lstm_3/while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
/model_3/lstm_3/while/lstm_cell_3/ReadVariableOpReadVariableOp:model_3_lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0
4model_3/lstm_3/while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
6model_3/lstm_3/while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  
6model_3/lstm_3/while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
.model_3/lstm_3/while/lstm_cell_3/strided_sliceStridedSlice7model_3/lstm_3/while/lstm_cell_3/ReadVariableOp:value:0=model_3/lstm_3/while/lstm_cell_3/strided_slice/stack:output:0?model_3/lstm_3/while/lstm_cell_3/strided_slice/stack_1:output:0?model_3/lstm_3/while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskË
)model_3/lstm_3/while/lstm_cell_3/MatMul_4MatMul*model_3/lstm_3/while/lstm_cell_3/mul_4:z:07model_3/lstm_3/while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬È
$model_3/lstm_3/while/lstm_cell_3/addAddV21model_3/lstm_3/while/lstm_cell_3/BiasAdd:output:03model_3/lstm_3/while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
(model_3/lstm_3/while/lstm_cell_3/SigmoidSigmoid(model_3/lstm_3/while/lstm_cell_3/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬®
1model_3/lstm_3/while/lstm_cell_3/ReadVariableOp_1ReadVariableOp:model_3_lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0
6model_3/lstm_3/while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  
8model_3/lstm_3/while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  
8model_3/lstm_3/while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
0model_3/lstm_3/while/lstm_cell_3/strided_slice_1StridedSlice9model_3/lstm_3/while/lstm_cell_3/ReadVariableOp_1:value:0?model_3/lstm_3/while/lstm_cell_3/strided_slice_1/stack:output:0Amodel_3/lstm_3/while/lstm_cell_3/strided_slice_1/stack_1:output:0Amodel_3/lstm_3/while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskÍ
)model_3/lstm_3/while/lstm_cell_3/MatMul_5MatMul*model_3/lstm_3/while/lstm_cell_3/mul_5:z:09model_3/lstm_3/while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ì
&model_3/lstm_3/while/lstm_cell_3/add_1AddV23model_3/lstm_3/while/lstm_cell_3/BiasAdd_1:output:03model_3/lstm_3/while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
*model_3/lstm_3/while/lstm_cell_3/Sigmoid_1Sigmoid*model_3/lstm_3/while/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬´
&model_3/lstm_3/while/lstm_cell_3/mul_8Mul.model_3/lstm_3/while/lstm_cell_3/Sigmoid_1:y:0"model_3_lstm_3_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬®
1model_3/lstm_3/while/lstm_cell_3/ReadVariableOp_2ReadVariableOp:model_3_lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0
6model_3/lstm_3/while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  
8model_3/lstm_3/while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
8model_3/lstm_3/while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
0model_3/lstm_3/while/lstm_cell_3/strided_slice_2StridedSlice9model_3/lstm_3/while/lstm_cell_3/ReadVariableOp_2:value:0?model_3/lstm_3/while/lstm_cell_3/strided_slice_2/stack:output:0Amodel_3/lstm_3/while/lstm_cell_3/strided_slice_2/stack_1:output:0Amodel_3/lstm_3/while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskÍ
)model_3/lstm_3/while/lstm_cell_3/MatMul_6MatMul*model_3/lstm_3/while/lstm_cell_3/mul_6:z:09model_3/lstm_3/while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ì
&model_3/lstm_3/while/lstm_cell_3/add_2AddV23model_3/lstm_3/while/lstm_cell_3/BiasAdd_2:output:03model_3/lstm_3/while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
%model_3/lstm_3/while/lstm_cell_3/TanhTanh*model_3/lstm_3/while/lstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¹
&model_3/lstm_3/while/lstm_cell_3/mul_9Mul,model_3/lstm_3/while/lstm_cell_3/Sigmoid:y:0)model_3/lstm_3/while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬º
&model_3/lstm_3/while/lstm_cell_3/add_3AddV2*model_3/lstm_3/while/lstm_cell_3/mul_8:z:0*model_3/lstm_3/while/lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬®
1model_3/lstm_3/while/lstm_cell_3/ReadVariableOp_3ReadVariableOp:model_3_lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0
6model_3/lstm_3/while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
8model_3/lstm_3/while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
8model_3/lstm_3/while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
0model_3/lstm_3/while/lstm_cell_3/strided_slice_3StridedSlice9model_3/lstm_3/while/lstm_cell_3/ReadVariableOp_3:value:0?model_3/lstm_3/while/lstm_cell_3/strided_slice_3/stack:output:0Amodel_3/lstm_3/while/lstm_cell_3/strided_slice_3/stack_1:output:0Amodel_3/lstm_3/while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskÍ
)model_3/lstm_3/while/lstm_cell_3/MatMul_7MatMul*model_3/lstm_3/while/lstm_cell_3/mul_7:z:09model_3/lstm_3/while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ì
&model_3/lstm_3/while/lstm_cell_3/add_4AddV23model_3/lstm_3/while/lstm_cell_3/BiasAdd_3:output:03model_3/lstm_3/while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
*model_3/lstm_3/while/lstm_cell_3/Sigmoid_2Sigmoid*model_3/lstm_3/while/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
'model_3/lstm_3/while/lstm_cell_3/Tanh_1Tanh*model_3/lstm_3/while/lstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¾
'model_3/lstm_3/while/lstm_cell_3/mul_10Mul.model_3/lstm_3/while/lstm_cell_3/Sigmoid_2:y:0+model_3/lstm_3/while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
9model_3/lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"model_3_lstm_3_while_placeholder_1 model_3_lstm_3_while_placeholder+model_3/lstm_3/while/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒ\
model_3/lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
model_3/lstm_3/while/addAddV2 model_3_lstm_3_while_placeholder#model_3/lstm_3/while/add/y:output:0*
T0*
_output_shapes
: ^
model_3/lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :£
model_3/lstm_3/while/add_1AddV26model_3_lstm_3_while_model_3_lstm_3_while_loop_counter%model_3/lstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: 
model_3/lstm_3/while/IdentityIdentitymodel_3/lstm_3/while/add_1:z:0^model_3/lstm_3/while/NoOp*
T0*
_output_shapes
: ¦
model_3/lstm_3/while/Identity_1Identity<model_3_lstm_3_while_model_3_lstm_3_while_maximum_iterations^model_3/lstm_3/while/NoOp*
T0*
_output_shapes
: 
model_3/lstm_3/while/Identity_2Identitymodel_3/lstm_3/while/add:z:0^model_3/lstm_3/while/NoOp*
T0*
_output_shapes
: Æ
model_3/lstm_3/while/Identity_3IdentityImodel_3/lstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^model_3/lstm_3/while/NoOp*
T0*
_output_shapes
: :éèÒ§
model_3/lstm_3/while/Identity_4Identity+model_3/lstm_3/while/lstm_cell_3/mul_10:z:0^model_3/lstm_3/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
model_3/lstm_3/while/Identity_5Identity*model_3/lstm_3/while/lstm_cell_3/add_3:z:0^model_3/lstm_3/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
model_3/lstm_3/while/NoOpNoOp0^model_3/lstm_3/while/lstm_cell_3/ReadVariableOp2^model_3/lstm_3/while/lstm_cell_3/ReadVariableOp_12^model_3/lstm_3/while/lstm_cell_3/ReadVariableOp_22^model_3/lstm_3/while/lstm_cell_3/ReadVariableOp_36^model_3/lstm_3/while/lstm_cell_3/split/ReadVariableOp8^model_3/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "G
model_3_lstm_3_while_identity&model_3/lstm_3/while/Identity:output:0"K
model_3_lstm_3_while_identity_1(model_3/lstm_3/while/Identity_1:output:0"K
model_3_lstm_3_while_identity_2(model_3/lstm_3/while/Identity_2:output:0"K
model_3_lstm_3_while_identity_3(model_3/lstm_3/while/Identity_3:output:0"K
model_3_lstm_3_while_identity_4(model_3/lstm_3/while/Identity_4:output:0"K
model_3_lstm_3_while_identity_5(model_3/lstm_3/while/Identity_5:output:0"v
8model_3_lstm_3_while_lstm_cell_3_readvariableop_resource:model_3_lstm_3_while_lstm_cell_3_readvariableop_resource_0"
@model_3_lstm_3_while_lstm_cell_3_split_1_readvariableop_resourceBmodel_3_lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0"
>model_3_lstm_3_while_lstm_cell_3_split_readvariableop_resource@model_3_lstm_3_while_lstm_cell_3_split_readvariableop_resource_0"h
1model_3_lstm_3_while_model_3_lstm_3_strided_slice3model_3_lstm_3_while_model_3_lstm_3_strided_slice_0"ä
omodel_3_lstm_3_while_tensorarrayv2read_tensorlistgetitem_model_3_lstm_3_tensorarrayunstack_tensorlistfromtensorqmodel_3_lstm_3_while_tensorarrayv2read_tensorlistgetitem_model_3_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : 2b
/model_3/lstm_3/while/lstm_cell_3/ReadVariableOp/model_3/lstm_3/while/lstm_cell_3/ReadVariableOp2f
1model_3/lstm_3/while/lstm_cell_3/ReadVariableOp_11model_3/lstm_3/while/lstm_cell_3/ReadVariableOp_12f
1model_3/lstm_3/while/lstm_cell_3/ReadVariableOp_21model_3/lstm_3/while/lstm_cell_3/ReadVariableOp_22f
1model_3/lstm_3/while/lstm_cell_3/ReadVariableOp_31model_3/lstm_3/while/lstm_cell_3/ReadVariableOp_32n
5model_3/lstm_3/while/lstm_cell_3/split/ReadVariableOp5model_3/lstm_3/while/lstm_cell_3/split/ReadVariableOp2r
7model_3/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp7model_3/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
: 
u
	
while_body_100315
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
1while_lstm_cell_3_split_readvariableop_resource_0:
È°	B
3while_lstm_cell_3_split_1_readvariableop_resource_0:	°	?
+while_lstm_cell_3_readvariableop_resource_0:
¬°	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
/while_lstm_cell_3_split_readvariableop_resource:
È°	@
1while_lstm_cell_3_split_1_readvariableop_resource:	°	=
)while_lstm_cell_3_readvariableop_resource:
¬°	¢ while/lstm_cell_3/ReadVariableOp¢"while/lstm_cell_3/ReadVariableOp_1¢"while/lstm_cell_3/ReadVariableOp_2¢"while/lstm_cell_3/ReadVariableOp_3¢&while/lstm_cell_3/split/ReadVariableOp¢(while/lstm_cell_3/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype0
!while/lstm_cell_3/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?®
while/lstm_cell_3/ones_likeFill*while/lstm_cell_3/ones_like/Shape:output:0*while/lstm_cell_3/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈf
#while/lstm_cell_3/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?´
while/lstm_cell_3/ones_like_1Fill,while/lstm_cell_3/ones_like_1/Shape:output:0,while/lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬§
while/lstm_cell_3/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ©
while/lstm_cell_3/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ©
while/lstm_cell_3/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ©
while/lstm_cell_3/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈc
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&while/lstm_cell_3/split/ReadVariableOpReadVariableOp1while_lstm_cell_3_split_readvariableop_resource_0* 
_output_shapes
:
È°	*
dtype0Ü
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0.while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È¬:
È¬:
È¬:
È¬*
	num_split
while/lstm_cell_3/MatMulMatMulwhile/lstm_cell_3/mul:z:0 while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/MatMul_1MatMulwhile/lstm_cell_3/mul_1:z:0 while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/MatMul_2MatMulwhile/lstm_cell_3/mul_2:z:0 while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/MatMul_3MatMulwhile/lstm_cell_3/mul_3:z:0 while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
#while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:°	*
dtype0Î
while/lstm_cell_3/split_1Split,while/lstm_cell_3/split_1/split_dim:output:00while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:¬:¬:¬:¬*
	num_split
while/lstm_cell_3/BiasAddBiasAdd"while/lstm_cell_3/MatMul:product:0"while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/lstm_cell_3/BiasAdd_1BiasAdd$while/lstm_cell_3/MatMul_1:product:0"while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/lstm_cell_3/BiasAdd_2BiasAdd$while/lstm_cell_3/MatMul_2:product:0"while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/lstm_cell_3/BiasAdd_3BiasAdd$while/lstm_cell_3/MatMul_3:product:0"while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_4Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_5Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_6Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_7Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 while/lstm_cell_3/ReadVariableOpReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0v
%while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  x
'while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell_3/strided_sliceStridedSlice(while/lstm_cell_3/ReadVariableOp:value:0.while/lstm_cell_3/strided_slice/stack:output:00while/lstm_cell_3/strided_slice/stack_1:output:00while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_4MatMulwhile/lstm_cell_3/mul_4:z:0(while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/addAddV2"while/lstm_cell_3/BiasAdd:output:0$while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r
while/lstm_cell_3/SigmoidSigmoidwhile/lstm_cell_3/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"while/lstm_cell_3/ReadVariableOp_1ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0x
'while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  z
)while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  z
)while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ñ
!while/lstm_cell_3/strided_slice_1StridedSlice*while/lstm_cell_3/ReadVariableOp_1:value:00while/lstm_cell_3/strided_slice_1/stack:output:02while/lstm_cell_3/strided_slice_1/stack_1:output:02while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask 
while/lstm_cell_3/MatMul_5MatMulwhile/lstm_cell_3/mul_5:z:0*while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/add_1AddV2$while/lstm_cell_3/BiasAdd_1:output:0$while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬v
while/lstm_cell_3/Sigmoid_1Sigmoidwhile/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_8Mulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"while/lstm_cell_3/ReadVariableOp_2ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0x
'while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  z
)while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ñ
!while/lstm_cell_3/strided_slice_2StridedSlice*while/lstm_cell_3/ReadVariableOp_2:value:00while/lstm_cell_3/strided_slice_2/stack:output:02while/lstm_cell_3/strided_slice_2/stack_1:output:02while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask 
while/lstm_cell_3/MatMul_6MatMulwhile/lstm_cell_3/mul_6:z:0*while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/add_2AddV2$while/lstm_cell_3/BiasAdd_2:output:0$while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬n
while/lstm_cell_3/TanhTanhwhile/lstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_9Mulwhile/lstm_cell_3/Sigmoid:y:0while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/add_3AddV2while/lstm_cell_3/mul_8:z:0while/lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"while/lstm_cell_3/ReadVariableOp_3ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0x
'while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ñ
!while/lstm_cell_3/strided_slice_3StridedSlice*while/lstm_cell_3/ReadVariableOp_3:value:00while/lstm_cell_3/strided_slice_3/stack:output:02while/lstm_cell_3/strided_slice_3/stack_1:output:02while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask 
while/lstm_cell_3/MatMul_7MatMulwhile/lstm_cell_3/mul_7:z:0*while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/add_4AddV2$while/lstm_cell_3/BiasAdd_3:output:0$while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬v
while/lstm_cell_3/Sigmoid_2Sigmoidwhile/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬p
while/lstm_cell_3/Tanh_1Tanhwhile/lstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_10Mulwhile/lstm_cell_3/Sigmoid_2:y:0while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_3/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬y
while/Identity_5Identitywhile/lstm_cell_3/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬²

while/NoOpNoOp!^while/lstm_cell_3/ReadVariableOp#^while/lstm_cell_3/ReadVariableOp_1#^while/lstm_cell_3/ReadVariableOp_2#^while/lstm_cell_3/ReadVariableOp_3'^while/lstm_cell_3/split/ReadVariableOp)^while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_3_readvariableop_resource+while_lstm_cell_3_readvariableop_resource_0"h
1while_lstm_cell_3_split_1_readvariableop_resource3while_lstm_cell_3_split_1_readvariableop_resource_0"d
/while_lstm_cell_3_split_readvariableop_resource1while_lstm_cell_3_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : 2D
 while/lstm_cell_3/ReadVariableOp while/lstm_cell_3/ReadVariableOp2H
"while/lstm_cell_3/ReadVariableOp_1"while/lstm_cell_3/ReadVariableOp_12H
"while/lstm_cell_3/ReadVariableOp_2"while/lstm_cell_3/ReadVariableOp_22H
"while/lstm_cell_3/ReadVariableOp_3"while/lstm_cell_3/ReadVariableOp_32P
&while/lstm_cell_3/split/ReadVariableOp&while/lstm_cell_3/split/ReadVariableOp2T
(while/lstm_cell_3/split_1/ReadVariableOp(while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
: 
Æ
ø
B__inference_model_3_layer_call_and_return_conditional_losses_99302

inputs
inputs_1
inputs_2
inputs_3%
embedding_1_99280:
¸dÈ 
lstm_3_99283:
È°	
lstm_3_99285:	°	 
lstm_3_99287:
¬°	*
time_distributed_99292:
¬¸d%
time_distributed_99294:	¸d
identity

identity_1

identity_2¢#embedding_1/StatefulPartitionedCall¢lstm_3/StatefulPartitionedCall¢(time_distributed/StatefulPartitionedCallõ
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1_99280*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_98565ì
lstm_3/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0inputs_2inputs_3lstm_3_99283lstm_3_99285lstm_3_99287*
Tin

2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_99224¿
(time_distributed/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0time_distributed_99292time_distributed_99294*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_98535o
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  ¨
time_distributed/ReshapeReshape'lstm_3/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
IdentityIdentity1time_distributed/StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸dy

Identity_1Identity'lstm_3/StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬y

Identity_2Identity'lstm_3/StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¸
NoOpNoOp$^embedding_1/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ>¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : : 2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:TP
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>¬
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
þt
	
while_body_98666
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
1while_lstm_cell_3_split_readvariableop_resource_0:
È°	B
3while_lstm_cell_3_split_1_readvariableop_resource_0:	°	?
+while_lstm_cell_3_readvariableop_resource_0:
¬°	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
/while_lstm_cell_3_split_readvariableop_resource:
È°	@
1while_lstm_cell_3_split_1_readvariableop_resource:	°	=
)while_lstm_cell_3_readvariableop_resource:
¬°	¢ while/lstm_cell_3/ReadVariableOp¢"while/lstm_cell_3/ReadVariableOp_1¢"while/lstm_cell_3/ReadVariableOp_2¢"while/lstm_cell_3/ReadVariableOp_3¢&while/lstm_cell_3/split/ReadVariableOp¢(while/lstm_cell_3/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype0
!while/lstm_cell_3/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?®
while/lstm_cell_3/ones_likeFill*while/lstm_cell_3/ones_like/Shape:output:0*while/lstm_cell_3/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈf
#while/lstm_cell_3/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?´
while/lstm_cell_3/ones_like_1Fill,while/lstm_cell_3/ones_like_1/Shape:output:0,while/lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬§
while/lstm_cell_3/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ©
while/lstm_cell_3/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ©
while/lstm_cell_3/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ©
while/lstm_cell_3/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈc
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&while/lstm_cell_3/split/ReadVariableOpReadVariableOp1while_lstm_cell_3_split_readvariableop_resource_0* 
_output_shapes
:
È°	*
dtype0Ü
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0.while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È¬:
È¬:
È¬:
È¬*
	num_split
while/lstm_cell_3/MatMulMatMulwhile/lstm_cell_3/mul:z:0 while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/MatMul_1MatMulwhile/lstm_cell_3/mul_1:z:0 while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/MatMul_2MatMulwhile/lstm_cell_3/mul_2:z:0 while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/MatMul_3MatMulwhile/lstm_cell_3/mul_3:z:0 while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
#while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:°	*
dtype0Î
while/lstm_cell_3/split_1Split,while/lstm_cell_3/split_1/split_dim:output:00while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:¬:¬:¬:¬*
	num_split
while/lstm_cell_3/BiasAddBiasAdd"while/lstm_cell_3/MatMul:product:0"while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/lstm_cell_3/BiasAdd_1BiasAdd$while/lstm_cell_3/MatMul_1:product:0"while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/lstm_cell_3/BiasAdd_2BiasAdd$while/lstm_cell_3/MatMul_2:product:0"while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/lstm_cell_3/BiasAdd_3BiasAdd$while/lstm_cell_3/MatMul_3:product:0"while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_4Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_5Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_6Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_7Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 while/lstm_cell_3/ReadVariableOpReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0v
%while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  x
'while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell_3/strided_sliceStridedSlice(while/lstm_cell_3/ReadVariableOp:value:0.while/lstm_cell_3/strided_slice/stack:output:00while/lstm_cell_3/strided_slice/stack_1:output:00while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_4MatMulwhile/lstm_cell_3/mul_4:z:0(while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/addAddV2"while/lstm_cell_3/BiasAdd:output:0$while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r
while/lstm_cell_3/SigmoidSigmoidwhile/lstm_cell_3/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"while/lstm_cell_3/ReadVariableOp_1ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0x
'while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  z
)while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  z
)while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ñ
!while/lstm_cell_3/strided_slice_1StridedSlice*while/lstm_cell_3/ReadVariableOp_1:value:00while/lstm_cell_3/strided_slice_1/stack:output:02while/lstm_cell_3/strided_slice_1/stack_1:output:02while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask 
while/lstm_cell_3/MatMul_5MatMulwhile/lstm_cell_3/mul_5:z:0*while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/add_1AddV2$while/lstm_cell_3/BiasAdd_1:output:0$while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬v
while/lstm_cell_3/Sigmoid_1Sigmoidwhile/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_8Mulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"while/lstm_cell_3/ReadVariableOp_2ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0x
'while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  z
)while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ñ
!while/lstm_cell_3/strided_slice_2StridedSlice*while/lstm_cell_3/ReadVariableOp_2:value:00while/lstm_cell_3/strided_slice_2/stack:output:02while/lstm_cell_3/strided_slice_2/stack_1:output:02while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask 
while/lstm_cell_3/MatMul_6MatMulwhile/lstm_cell_3/mul_6:z:0*while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/add_2AddV2$while/lstm_cell_3/BiasAdd_2:output:0$while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬n
while/lstm_cell_3/TanhTanhwhile/lstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_9Mulwhile/lstm_cell_3/Sigmoid:y:0while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/add_3AddV2while/lstm_cell_3/mul_8:z:0while/lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"while/lstm_cell_3/ReadVariableOp_3ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0x
'while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ñ
!while/lstm_cell_3/strided_slice_3StridedSlice*while/lstm_cell_3/ReadVariableOp_3:value:00while/lstm_cell_3/strided_slice_3/stack:output:02while/lstm_cell_3/strided_slice_3/stack_1:output:02while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask 
while/lstm_cell_3/MatMul_7MatMulwhile/lstm_cell_3/mul_7:z:0*while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/add_4AddV2$while/lstm_cell_3/BiasAdd_3:output:0$while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬v
while/lstm_cell_3/Sigmoid_2Sigmoidwhile/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬p
while/lstm_cell_3/Tanh_1Tanhwhile/lstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_10Mulwhile/lstm_cell_3/Sigmoid_2:y:0while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_3/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬y
while/Identity_5Identitywhile/lstm_cell_3/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬²

while/NoOpNoOp!^while/lstm_cell_3/ReadVariableOp#^while/lstm_cell_3/ReadVariableOp_1#^while/lstm_cell_3/ReadVariableOp_2#^while/lstm_cell_3/ReadVariableOp_3'^while/lstm_cell_3/split/ReadVariableOp)^while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_3_readvariableop_resource+while_lstm_cell_3_readvariableop_resource_0"h
1while_lstm_cell_3_split_1_readvariableop_resource3while_lstm_cell_3_split_1_readvariableop_resource_0"d
/while_lstm_cell_3_split_readvariableop_resource1while_lstm_cell_3_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : 2D
 while/lstm_cell_3/ReadVariableOp while/lstm_cell_3/ReadVariableOp2H
"while/lstm_cell_3/ReadVariableOp_1"while/lstm_cell_3/ReadVariableOp_12H
"while/lstm_cell_3/ReadVariableOp_2"while/lstm_cell_3/ReadVariableOp_22H
"while/lstm_cell_3/ReadVariableOp_3"while/lstm_cell_3/ReadVariableOp_32P
&while/lstm_cell_3/split/ReadVariableOp&while/lstm_cell_3/split/ReadVariableOp2T
(while/lstm_cell_3/split_1/ReadVariableOp(while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
: 
óø
º
C__inference_model_3_layer_call_and_return_conditional_losses_100099
inputs_0
inputs_1
inputs_2
inputs_36
"embedding_1_embedding_lookup_99717:
¸dÈD
0lstm_3_lstm_cell_3_split_readvariableop_resource:
È°	A
2lstm_3_lstm_cell_3_split_1_readvariableop_resource:	°	>
*lstm_3_lstm_cell_3_readvariableop_resource:
¬°	I
5time_distributed_dense_matmul_readvariableop_resource:
¬¸dE
6time_distributed_dense_biasadd_readvariableop_resource:	¸d
identity

identity_1

identity_2¢embedding_1/embedding_lookup¢!lstm_3/lstm_cell_3/ReadVariableOp¢#lstm_3/lstm_cell_3/ReadVariableOp_1¢#lstm_3/lstm_cell_3/ReadVariableOp_2¢#lstm_3/lstm_cell_3/ReadVariableOp_3¢'lstm_3/lstm_cell_3/split/ReadVariableOp¢)lstm_3/lstm_cell_3/split_1/ReadVariableOp¢lstm_3/while¢-time_distributed/dense/BiasAdd/ReadVariableOp¢,time_distributed/dense/MatMul/ReadVariableOpl
embedding_1/CastCastinputs_0*

DstT0*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿó
embedding_1/embedding_lookupResourceGather"embedding_1_embedding_lookup_99717embedding_1/Cast:y:0*
Tindices0*5
_class+
)'loc:@embedding_1/embedding_lookup/99717*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*
dtype0Ï
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_1/embedding_lookup/99717*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ£
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈj
lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¯
lstm_3/transpose	Transpose0embedding_1/embedding_lookup/Identity_1:output:0lstm_3/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈP
lstm_3/ShapeShapelstm_3/transpose:y:0*
T0*
_output_shapes
:d
lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm_3/strided_sliceStridedSlicelstm_3/Shape:output:0#lstm_3/strided_slice/stack:output:0%lstm_3/strided_slice/stack_1:output:0%lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
lstm_3/TensorArrayV2TensorListReserve+lstm_3/TensorArrayV2/element_shape:output:0lstm_3/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   õ
.lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_3/transpose:y:0Elstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒf
lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_3/strided_slice_1StridedSlicelstm_3/transpose:y:0%lstm_3/strided_slice_1/stack:output:0'lstm_3/strided_slice_1/stack_1:output:0'lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_maskq
"lstm_3/lstm_cell_3/ones_like/ShapeShapelstm_3/strided_slice_1:output:0*
T0*
_output_shapes
:g
"lstm_3/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?±
lstm_3/lstm_cell_3/ones_likeFill+lstm_3/lstm_cell_3/ones_like/Shape:output:0+lstm_3/lstm_cell_3/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈe
 lstm_3/lstm_cell_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?ª
lstm_3/lstm_cell_3/dropout/MulMul%lstm_3/lstm_cell_3/ones_like:output:0)lstm_3/lstm_cell_3/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈu
 lstm_3/lstm_cell_3/dropout/ShapeShape%lstm_3/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:³
7lstm_3/lstm_cell_3/dropout/random_uniform/RandomUniformRandomUniform)lstm_3/lstm_cell_3/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0n
)lstm_3/lstm_cell_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>à
'lstm_3/lstm_cell_3/dropout/GreaterEqualGreaterEqual@lstm_3/lstm_cell_3/dropout/random_uniform/RandomUniform:output:02lstm_3/lstm_cell_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_3/lstm_cell_3/dropout/CastCast+lstm_3/lstm_cell_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ£
 lstm_3/lstm_cell_3/dropout/Mul_1Mul"lstm_3/lstm_cell_3/dropout/Mul:z:0#lstm_3/lstm_cell_3/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈg
"lstm_3/lstm_cell_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?®
 lstm_3/lstm_cell_3/dropout_1/MulMul%lstm_3/lstm_cell_3/ones_like:output:0+lstm_3/lstm_cell_3/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈw
"lstm_3/lstm_cell_3/dropout_1/ShapeShape%lstm_3/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:·
9lstm_3/lstm_cell_3/dropout_1/random_uniform/RandomUniformRandomUniform+lstm_3/lstm_cell_3/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0p
+lstm_3/lstm_cell_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>æ
)lstm_3/lstm_cell_3/dropout_1/GreaterEqualGreaterEqualBlstm_3/lstm_cell_3/dropout_1/random_uniform/RandomUniform:output:04lstm_3/lstm_cell_3/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
!lstm_3/lstm_cell_3/dropout_1/CastCast-lstm_3/lstm_cell_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ©
"lstm_3/lstm_cell_3/dropout_1/Mul_1Mul$lstm_3/lstm_cell_3/dropout_1/Mul:z:0%lstm_3/lstm_cell_3/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈg
"lstm_3/lstm_cell_3/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?®
 lstm_3/lstm_cell_3/dropout_2/MulMul%lstm_3/lstm_cell_3/ones_like:output:0+lstm_3/lstm_cell_3/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈw
"lstm_3/lstm_cell_3/dropout_2/ShapeShape%lstm_3/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:·
9lstm_3/lstm_cell_3/dropout_2/random_uniform/RandomUniformRandomUniform+lstm_3/lstm_cell_3/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0p
+lstm_3/lstm_cell_3/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>æ
)lstm_3/lstm_cell_3/dropout_2/GreaterEqualGreaterEqualBlstm_3/lstm_cell_3/dropout_2/random_uniform/RandomUniform:output:04lstm_3/lstm_cell_3/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
!lstm_3/lstm_cell_3/dropout_2/CastCast-lstm_3/lstm_cell_3/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ©
"lstm_3/lstm_cell_3/dropout_2/Mul_1Mul$lstm_3/lstm_cell_3/dropout_2/Mul:z:0%lstm_3/lstm_cell_3/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈg
"lstm_3/lstm_cell_3/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?®
 lstm_3/lstm_cell_3/dropout_3/MulMul%lstm_3/lstm_cell_3/ones_like:output:0+lstm_3/lstm_cell_3/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈw
"lstm_3/lstm_cell_3/dropout_3/ShapeShape%lstm_3/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:·
9lstm_3/lstm_cell_3/dropout_3/random_uniform/RandomUniformRandomUniform+lstm_3/lstm_cell_3/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0p
+lstm_3/lstm_cell_3/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>æ
)lstm_3/lstm_cell_3/dropout_3/GreaterEqualGreaterEqualBlstm_3/lstm_cell_3/dropout_3/random_uniform/RandomUniform:output:04lstm_3/lstm_cell_3/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
!lstm_3/lstm_cell_3/dropout_3/CastCast-lstm_3/lstm_cell_3/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ©
"lstm_3/lstm_cell_3/dropout_3/Mul_1Mul$lstm_3/lstm_cell_3/dropout_3/Mul:z:0%lstm_3/lstm_cell_3/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ\
$lstm_3/lstm_cell_3/ones_like_1/ShapeShapeinputs_2*
T0*
_output_shapes
:i
$lstm_3/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?·
lstm_3/lstm_cell_3/ones_like_1Fill-lstm_3/lstm_cell_3/ones_like_1/Shape:output:0-lstm_3/lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬g
"lstm_3/lstm_cell_3/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?°
 lstm_3/lstm_cell_3/dropout_4/MulMul'lstm_3/lstm_cell_3/ones_like_1:output:0+lstm_3/lstm_cell_3/dropout_4/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬y
"lstm_3/lstm_cell_3/dropout_4/ShapeShape'lstm_3/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:·
9lstm_3/lstm_cell_3/dropout_4/random_uniform/RandomUniformRandomUniform+lstm_3/lstm_cell_3/dropout_4/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0p
+lstm_3/lstm_cell_3/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>æ
)lstm_3/lstm_cell_3/dropout_4/GreaterEqualGreaterEqualBlstm_3/lstm_cell_3/dropout_4/random_uniform/RandomUniform:output:04lstm_3/lstm_cell_3/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
!lstm_3/lstm_cell_3/dropout_4/CastCast-lstm_3/lstm_cell_3/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬©
"lstm_3/lstm_cell_3/dropout_4/Mul_1Mul$lstm_3/lstm_cell_3/dropout_4/Mul:z:0%lstm_3/lstm_cell_3/dropout_4/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬g
"lstm_3/lstm_cell_3/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?°
 lstm_3/lstm_cell_3/dropout_5/MulMul'lstm_3/lstm_cell_3/ones_like_1:output:0+lstm_3/lstm_cell_3/dropout_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬y
"lstm_3/lstm_cell_3/dropout_5/ShapeShape'lstm_3/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:·
9lstm_3/lstm_cell_3/dropout_5/random_uniform/RandomUniformRandomUniform+lstm_3/lstm_cell_3/dropout_5/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0p
+lstm_3/lstm_cell_3/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>æ
)lstm_3/lstm_cell_3/dropout_5/GreaterEqualGreaterEqualBlstm_3/lstm_cell_3/dropout_5/random_uniform/RandomUniform:output:04lstm_3/lstm_cell_3/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
!lstm_3/lstm_cell_3/dropout_5/CastCast-lstm_3/lstm_cell_3/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬©
"lstm_3/lstm_cell_3/dropout_5/Mul_1Mul$lstm_3/lstm_cell_3/dropout_5/Mul:z:0%lstm_3/lstm_cell_3/dropout_5/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬g
"lstm_3/lstm_cell_3/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?°
 lstm_3/lstm_cell_3/dropout_6/MulMul'lstm_3/lstm_cell_3/ones_like_1:output:0+lstm_3/lstm_cell_3/dropout_6/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬y
"lstm_3/lstm_cell_3/dropout_6/ShapeShape'lstm_3/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:·
9lstm_3/lstm_cell_3/dropout_6/random_uniform/RandomUniformRandomUniform+lstm_3/lstm_cell_3/dropout_6/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0p
+lstm_3/lstm_cell_3/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>æ
)lstm_3/lstm_cell_3/dropout_6/GreaterEqualGreaterEqualBlstm_3/lstm_cell_3/dropout_6/random_uniform/RandomUniform:output:04lstm_3/lstm_cell_3/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
!lstm_3/lstm_cell_3/dropout_6/CastCast-lstm_3/lstm_cell_3/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬©
"lstm_3/lstm_cell_3/dropout_6/Mul_1Mul$lstm_3/lstm_cell_3/dropout_6/Mul:z:0%lstm_3/lstm_cell_3/dropout_6/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬g
"lstm_3/lstm_cell_3/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?°
 lstm_3/lstm_cell_3/dropout_7/MulMul'lstm_3/lstm_cell_3/ones_like_1:output:0+lstm_3/lstm_cell_3/dropout_7/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬y
"lstm_3/lstm_cell_3/dropout_7/ShapeShape'lstm_3/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:·
9lstm_3/lstm_cell_3/dropout_7/random_uniform/RandomUniformRandomUniform+lstm_3/lstm_cell_3/dropout_7/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0p
+lstm_3/lstm_cell_3/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>æ
)lstm_3/lstm_cell_3/dropout_7/GreaterEqualGreaterEqualBlstm_3/lstm_cell_3/dropout_7/random_uniform/RandomUniform:output:04lstm_3/lstm_cell_3/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
!lstm_3/lstm_cell_3/dropout_7/CastCast-lstm_3/lstm_cell_3/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬©
"lstm_3/lstm_cell_3/dropout_7/Mul_1Mul$lstm_3/lstm_cell_3/dropout_7/Mul:z:0%lstm_3/lstm_cell_3/dropout_7/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/lstm_cell_3/mulMullstm_3/strided_slice_1:output:0$lstm_3/lstm_cell_3/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_3/lstm_cell_3/mul_1Mullstm_3/strided_slice_1:output:0&lstm_3/lstm_cell_3/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_3/lstm_cell_3/mul_2Mullstm_3/strided_slice_1:output:0&lstm_3/lstm_cell_3/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_3/lstm_cell_3/mul_3Mullstm_3/strided_slice_1:output:0&lstm_3/lstm_cell_3/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈd
"lstm_3/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
'lstm_3/lstm_cell_3/split/ReadVariableOpReadVariableOp0lstm_3_lstm_cell_3_split_readvariableop_resource* 
_output_shapes
:
È°	*
dtype0ß
lstm_3/lstm_cell_3/splitSplit+lstm_3/lstm_cell_3/split/split_dim:output:0/lstm_3/lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È¬:
È¬:
È¬:
È¬*
	num_split
lstm_3/lstm_cell_3/MatMulMatMullstm_3/lstm_cell_3/mul:z:0!lstm_3/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/lstm_cell_3/MatMul_1MatMullstm_3/lstm_cell_3/mul_1:z:0!lstm_3/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/lstm_cell_3/MatMul_2MatMullstm_3/lstm_cell_3/mul_2:z:0!lstm_3/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/lstm_cell_3/MatMul_3MatMullstm_3/lstm_cell_3/mul_3:z:0!lstm_3/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
$lstm_3/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
)lstm_3/lstm_cell_3/split_1/ReadVariableOpReadVariableOp2lstm_3_lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:°	*
dtype0Ñ
lstm_3/lstm_cell_3/split_1Split-lstm_3/lstm_cell_3/split_1/split_dim:output:01lstm_3/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:¬:¬:¬:¬*
	num_split¢
lstm_3/lstm_cell_3/BiasAddBiasAdd#lstm_3/lstm_cell_3/MatMul:product:0#lstm_3/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
lstm_3/lstm_cell_3/BiasAdd_1BiasAdd%lstm_3/lstm_cell_3/MatMul_1:product:0#lstm_3/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
lstm_3/lstm_cell_3/BiasAdd_2BiasAdd%lstm_3/lstm_cell_3/MatMul_2:product:0#lstm_3/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
lstm_3/lstm_cell_3/BiasAdd_3BiasAdd%lstm_3/lstm_cell_3/MatMul_3:product:0#lstm_3/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/lstm_cell_3/mul_4Mulinputs_2&lstm_3/lstm_cell_3/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/lstm_cell_3/mul_5Mulinputs_2&lstm_3/lstm_cell_3/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/lstm_cell_3/mul_6Mulinputs_2&lstm_3/lstm_cell_3/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/lstm_cell_3/mul_7Mulinputs_2&lstm_3/lstm_cell_3/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
!lstm_3/lstm_cell_3/ReadVariableOpReadVariableOp*lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0w
&lstm_3/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(lstm_3/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  y
(lstm_3/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ì
 lstm_3/lstm_cell_3/strided_sliceStridedSlice)lstm_3/lstm_cell_3/ReadVariableOp:value:0/lstm_3/lstm_cell_3/strided_slice/stack:output:01lstm_3/lstm_cell_3/strided_slice/stack_1:output:01lstm_3/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask¡
lstm_3/lstm_cell_3/MatMul_4MatMullstm_3/lstm_cell_3/mul_4:z:0)lstm_3/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/lstm_cell_3/addAddV2#lstm_3/lstm_cell_3/BiasAdd:output:0%lstm_3/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬t
lstm_3/lstm_cell_3/SigmoidSigmoidlstm_3/lstm_cell_3/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
#lstm_3/lstm_cell_3/ReadVariableOp_1ReadVariableOp*lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0y
(lstm_3/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  {
*lstm_3/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  {
*lstm_3/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm_3/lstm_cell_3/strided_slice_1StridedSlice+lstm_3/lstm_cell_3/ReadVariableOp_1:value:01lstm_3/lstm_cell_3/strided_slice_1/stack:output:03lstm_3/lstm_cell_3/strided_slice_1/stack_1:output:03lstm_3/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask£
lstm_3/lstm_cell_3/MatMul_5MatMullstm_3/lstm_cell_3/mul_5:z:0+lstm_3/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¢
lstm_3/lstm_cell_3/add_1AddV2%lstm_3/lstm_cell_3/BiasAdd_1:output:0%lstm_3/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬x
lstm_3/lstm_cell_3/Sigmoid_1Sigmoidlstm_3/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬~
lstm_3/lstm_cell_3/mul_8Mul lstm_3/lstm_cell_3/Sigmoid_1:y:0inputs_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
#lstm_3/lstm_cell_3/ReadVariableOp_2ReadVariableOp*lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0y
(lstm_3/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  {
*lstm_3/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      {
*lstm_3/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm_3/lstm_cell_3/strided_slice_2StridedSlice+lstm_3/lstm_cell_3/ReadVariableOp_2:value:01lstm_3/lstm_cell_3/strided_slice_2/stack:output:03lstm_3/lstm_cell_3/strided_slice_2/stack_1:output:03lstm_3/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask£
lstm_3/lstm_cell_3/MatMul_6MatMullstm_3/lstm_cell_3/mul_6:z:0+lstm_3/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¢
lstm_3/lstm_cell_3/add_2AddV2%lstm_3/lstm_cell_3/BiasAdd_2:output:0%lstm_3/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬p
lstm_3/lstm_cell_3/TanhTanhlstm_3/lstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/lstm_cell_3/mul_9Mullstm_3/lstm_cell_3/Sigmoid:y:0lstm_3/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/lstm_cell_3/add_3AddV2lstm_3/lstm_cell_3/mul_8:z:0lstm_3/lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
#lstm_3/lstm_cell_3/ReadVariableOp_3ReadVariableOp*lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0y
(lstm_3/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      {
*lstm_3/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_3/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm_3/lstm_cell_3/strided_slice_3StridedSlice+lstm_3/lstm_cell_3/ReadVariableOp_3:value:01lstm_3/lstm_cell_3/strided_slice_3/stack:output:03lstm_3/lstm_cell_3/strided_slice_3/stack_1:output:03lstm_3/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask£
lstm_3/lstm_cell_3/MatMul_7MatMullstm_3/lstm_cell_3/mul_7:z:0+lstm_3/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¢
lstm_3/lstm_cell_3/add_4AddV2%lstm_3/lstm_cell_3/BiasAdd_3:output:0%lstm_3/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬x
lstm_3/lstm_cell_3/Sigmoid_2Sigmoidlstm_3/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r
lstm_3/lstm_cell_3/Tanh_1Tanhlstm_3/lstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/lstm_cell_3/mul_10Mul lstm_3/lstm_cell_3/Sigmoid_2:y:0lstm_3/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬u
$lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  Ë
lstm_3/TensorArrayV2_1TensorListReserve-lstm_3/TensorArrayV2_1/element_shape:output:0lstm_3/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒM
lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : »
lstm_3/whileWhile"lstm_3/while/loop_counter:output:0(lstm_3/while/maximum_iterations:output:0lstm_3/time:output:0lstm_3/TensorArrayV2_1:handle:0inputs_2inputs_3lstm_3/strided_slice:output:0>lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:00lstm_3_lstm_cell_3_split_readvariableop_resource2lstm_3_lstm_cell_3_split_1_readvariableop_resource*lstm_3_lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_3_while_body_99879*#
condR
lstm_3_while_cond_99878*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *
parallel_iterations 
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  á
)lstm_3/TensorArrayV2Stack/TensorListStackTensorListStacklstm_3/while:output:3@lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬*
element_dtype0o
lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿh
lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
lstm_3/strided_slice_2StridedSlice2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_3/strided_slice_2/stack:output:0'lstm_3/strided_slice_2/stack_1:output:0'lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_maskl
lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          µ
lstm_3/transpose_1	Transpose2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_3/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬b
lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    \
time_distributed/ShapeShapelstm_3/transpose_1:y:0*
T0*
_output_shapes
:n
$time_distributed/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:p
&time_distributed/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&time_distributed/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
time_distributed/strided_sliceStridedSlicetime_distributed/Shape:output:0-time_distributed/strided_slice/stack:output:0/time_distributed/strided_slice/stack_1:output:0/time_distributed/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  
time_distributed/ReshapeReshapelstm_3/transpose_1:y:0'time_distributed/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¤
,time_distributed/dense/MatMul/ReadVariableOpReadVariableOp5time_distributed_dense_matmul_readvariableop_resource* 
_output_shapes
:
¬¸d*
dtype0³
time_distributed/dense/MatMulMatMul!time_distributed/Reshape:output:04time_distributed/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸d¡
-time_distributed/dense/BiasAdd/ReadVariableOpReadVariableOp6time_distributed_dense_biasadd_readvariableop_resource*
_output_shapes	
:¸d*
dtype0¼
time_distributed/dense/BiasAddBiasAdd'time_distributed/dense/MatMul:product:05time_distributed/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸d
time_distributed/dense/SoftmaxSoftmax'time_distributed/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸dm
"time_distributed/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿe
"time_distributed/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :¸dÙ
 time_distributed/Reshape_1/shapePack+time_distributed/Reshape_1/shape/0:output:0'time_distributed/strided_slice:output:0+time_distributed/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:º
time_distributed/Reshape_1Reshape(time_distributed/dense/Softmax:softmax:0)time_distributed/Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸dq
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  
time_distributed/Reshape_2Reshapelstm_3/transpose_1:y:0)time_distributed/Reshape_2/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
IdentityIdentity#time_distributed/Reshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸dg

Identity_1Identitylstm_3/while:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬g

Identity_2Identitylstm_3/while:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¿
NoOpNoOp^embedding_1/embedding_lookup"^lstm_3/lstm_cell_3/ReadVariableOp$^lstm_3/lstm_cell_3/ReadVariableOp_1$^lstm_3/lstm_cell_3/ReadVariableOp_2$^lstm_3/lstm_cell_3/ReadVariableOp_3(^lstm_3/lstm_cell_3/split/ReadVariableOp*^lstm_3/lstm_cell_3/split_1/ReadVariableOp^lstm_3/while.^time_distributed/dense/BiasAdd/ReadVariableOp-^time_distributed/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ>¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : : 2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2F
!lstm_3/lstm_cell_3/ReadVariableOp!lstm_3/lstm_cell_3/ReadVariableOp2J
#lstm_3/lstm_cell_3/ReadVariableOp_1#lstm_3/lstm_cell_3/ReadVariableOp_12J
#lstm_3/lstm_cell_3/ReadVariableOp_2#lstm_3/lstm_cell_3/ReadVariableOp_22J
#lstm_3/lstm_cell_3/ReadVariableOp_3#lstm_3/lstm_cell_3/ReadVariableOp_32R
'lstm_3/lstm_cell_3/split/ReadVariableOp'lstm_3/lstm_cell_3/split/ReadVariableOp2V
)lstm_3/lstm_cell_3/split_1/ReadVariableOp)lstm_3/lstm_cell_3/split_1/ReadVariableOp2
lstm_3/whilelstm_3/while2^
-time_distributed/dense/BiasAdd/ReadVariableOp-time_distributed/dense/BiasAdd/ReadVariableOp2\
,time_distributed/dense/MatMul/ReadVariableOp,time_distributed/dense/MatMul/ReadVariableOp:Z V
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>¬
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
inputs/2:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
inputs/3
¨

ô
@__inference_dense_layer_call_and_return_conditional_losses_98485

inputs2
matmul_readvariableop_resource:
¬¸d.
biasadd_readvariableop_resource:	¸d
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¬¸d*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸ds
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¸d*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸dW
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸da
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
ÕD
©
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_98051

inputs

states
states_11
split_readvariableop_resource:
È°	.
split_1_readvariableop_resource:	°	+
readvariableop_resource:
¬°	
identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈG
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Y
mulMulinputsones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ[
mul_1Mulinputsones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ[
mul_2Mulinputsones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ[
mul_3Mulinputsones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
È°	*
dtype0¦
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È¬:
È¬:
È¬:
È¬*
	num_split\
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:°	*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:¬:¬:¬:¬*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬]
mul_4Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬]
mul_5Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬]
mul_6Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬]
mul_7Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
¬°	*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskh
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
¬°	*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskj
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬X
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
¬°	*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskj
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬V
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬W
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
¬°	*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskj
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬R
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬L
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬[
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
IdentityIdentity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬\

Identity_1Identity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬[

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬À
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_namestates
°
¼
while_cond_98665
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_98665___redundant_placeholder03
/while_while_cond_98665___redundant_placeholder13
/while_while_cond_98665___redundant_placeholder23
/while_while_cond_98665___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
:
¹
Ã
while_cond_100623
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_100623___redundant_placeholder04
0while_while_cond_100623___redundant_placeholder14
0while_while_cond_100623___redundant_placeholder24
0while_while_cond_100623___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
:
µ
Á
while_cond_101219
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_101219___redundant_placeholder04
0while_while_cond_101219___redundant_placeholder14
0while_while_cond_101219___redundant_placeholder24
0while_while_cond_101219___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
:
ä	
¤
F__inference_embedding_1_layer_call_and_return_conditional_losses_98565

inputs*
embedding_lookup_98559:
¸dÈ
identity¢embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÃ
embedding_lookupResourceGatherembedding_lookup_98559Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/98559*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*
dtype0«
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/98559*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹
Ã
while_cond_100314
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_100314___redundant_placeholder04
0while_while_cond_100314___redundant_placeholder14
0while_while_cond_100314___redundant_placeholder24
0while_while_cond_100314___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
:
ô	
È
lstm_3_while_cond_99878*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3*
&lstm_3_while_less_lstm_3_strided_sliceA
=lstm_3_while_lstm_3_while_cond_99878___redundant_placeholder0A
=lstm_3_while_lstm_3_while_cond_99878___redundant_placeholder1A
=lstm_3_while_lstm_3_while_cond_99878___redundant_placeholder2A
=lstm_3_while_lstm_3_while_cond_99878___redundant_placeholder3
lstm_3_while_identity
|
lstm_3/while/LessLesslstm_3_while_placeholder&lstm_3_while_less_lstm_3_strided_slice*
T0*
_output_shapes
: Y
lstm_3/while/IdentityIdentitylstm_3/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_3_while_identitylstm_3/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
:
Â
¦
A__inference_lstm_3_layer_call_and_return_conditional_losses_99224

inputs
initial_state
initial_state_1=
)lstm_cell_3_split_readvariableop_resource:
È°	:
+lstm_cell_3_split_1_readvariableop_resource:	°	7
#lstm_cell_3_readvariableop_resource:
¬°	
identity

identity_1

identity_2¢lstm_cell_3/ReadVariableOp¢lstm_cell_3/ReadVariableOp_1¢lstm_cell_3/ReadVariableOp_2¢lstm_cell_3/ReadVariableOp_3¢ lstm_cell_3/split/ReadVariableOp¢"lstm_cell_3/split_1/ReadVariableOp¢whilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈB
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_maskc
lstm_cell_3/ones_like/ShapeShapestrided_slice_1:output:0*
T0*
_output_shapes
:`
lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_3/ones_likeFill$lstm_cell_3/ones_like/Shape:output:0$lstm_cell_3/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ^
lstm_cell_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?
lstm_cell_3/dropout/MulMullstm_cell_3/ones_like:output:0"lstm_cell_3/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈg
lstm_cell_3/dropout/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:¥
0lstm_cell_3/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_3/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0g
"lstm_cell_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>Ë
 lstm_cell_3/dropout/GreaterEqualGreaterEqual9lstm_cell_3/dropout/random_uniform/RandomUniform:output:0+lstm_cell_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/dropout/CastCast$lstm_cell_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/dropout/Mul_1Mullstm_cell_3/dropout/Mul:z:0lstm_cell_3/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ`
lstm_cell_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?
lstm_cell_3/dropout_1/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈi
lstm_cell_3/dropout_1/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:©
2lstm_cell_3/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0i
$lstm_cell_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>Ñ
"lstm_cell_3/dropout_1/GreaterEqualGreaterEqual;lstm_cell_3/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/dropout_1/CastCast&lstm_cell_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/dropout_1/Mul_1Mullstm_cell_3/dropout_1/Mul:z:0lstm_cell_3/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ`
lstm_cell_3/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?
lstm_cell_3/dropout_2/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈi
lstm_cell_3/dropout_2/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:©
2lstm_cell_3/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0i
$lstm_cell_3/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>Ñ
"lstm_cell_3/dropout_2/GreaterEqualGreaterEqual;lstm_cell_3/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/dropout_2/CastCast&lstm_cell_3/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/dropout_2/Mul_1Mullstm_cell_3/dropout_2/Mul:z:0lstm_cell_3/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ`
lstm_cell_3/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?
lstm_cell_3/dropout_3/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈi
lstm_cell_3/dropout_3/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:©
2lstm_cell_3/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0i
$lstm_cell_3/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>Ñ
"lstm_cell_3/dropout_3/GreaterEqualGreaterEqual;lstm_cell_3/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/dropout_3/CastCast&lstm_cell_3/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/dropout_3/Mul_1Mullstm_cell_3/dropout_3/Mul:z:0lstm_cell_3/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈZ
lstm_cell_3/ones_like_1/ShapeShapeinitial_state*
T0*
_output_shapes
:b
lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¢
lstm_cell_3/ones_like_1Fill&lstm_cell_3/ones_like_1/Shape:output:0&lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
lstm_cell_3/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_3/dropout_4/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_4/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
lstm_cell_3/dropout_4/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:©
2lstm_cell_3/dropout_4/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_4/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0i
$lstm_cell_3/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ñ
"lstm_cell_3/dropout_4/GreaterEqualGreaterEqual;lstm_cell_3/dropout_4/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/dropout_4/CastCast&lstm_cell_3/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/dropout_4/Mul_1Mullstm_cell_3/dropout_4/Mul:z:0lstm_cell_3/dropout_4/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
lstm_cell_3/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_3/dropout_5/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
lstm_cell_3/dropout_5/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:©
2lstm_cell_3/dropout_5/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_5/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0i
$lstm_cell_3/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ñ
"lstm_cell_3/dropout_5/GreaterEqualGreaterEqual;lstm_cell_3/dropout_5/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/dropout_5/CastCast&lstm_cell_3/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/dropout_5/Mul_1Mullstm_cell_3/dropout_5/Mul:z:0lstm_cell_3/dropout_5/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
lstm_cell_3/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_3/dropout_6/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_6/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
lstm_cell_3/dropout_6/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:©
2lstm_cell_3/dropout_6/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_6/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0i
$lstm_cell_3/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ñ
"lstm_cell_3/dropout_6/GreaterEqualGreaterEqual;lstm_cell_3/dropout_6/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/dropout_6/CastCast&lstm_cell_3/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/dropout_6/Mul_1Mullstm_cell_3/dropout_6/Mul:z:0lstm_cell_3/dropout_6/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
lstm_cell_3/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_3/dropout_7/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_7/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
lstm_cell_3/dropout_7/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:©
2lstm_cell_3/dropout_7/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_7/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0i
$lstm_cell_3/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ñ
"lstm_cell_3/dropout_7/GreaterEqualGreaterEqual;lstm_cell_3/dropout_7/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/dropout_7/CastCast&lstm_cell_3/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/dropout_7/Mul_1Mullstm_cell_3/dropout_7/Mul:z:0lstm_cell_3/dropout_7/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/mulMulstrided_slice_1:output:0lstm_cell_3/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/mul_1Mulstrided_slice_1:output:0lstm_cell_3/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/mul_2Mulstrided_slice_1:output:0lstm_cell_3/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/mul_3Mulstrided_slice_1:output:0lstm_cell_3/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ]
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_cell_3/split/ReadVariableOpReadVariableOp)lstm_cell_3_split_readvariableop_resource* 
_output_shapes
:
È°	*
dtype0Ê
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0(lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È¬:
È¬:
È¬:
È¬*
	num_split
lstm_cell_3/MatMulMatMullstm_cell_3/mul:z:0lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/MatMul_1MatMullstm_cell_3/mul_1:z:0lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/MatMul_2MatMullstm_cell_3/mul_2:z:0lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/MatMul_3MatMullstm_cell_3/mul_3:z:0lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬_
lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"lstm_cell_3/split_1/ReadVariableOpReadVariableOp+lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:°	*
dtype0¼
lstm_cell_3/split_1Split&lstm_cell_3/split_1/split_dim:output:0*lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:¬:¬:¬:¬*
	num_split
lstm_cell_3/BiasAddBiasAddlstm_cell_3/MatMul:product:0lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/BiasAdd_1BiasAddlstm_cell_3/MatMul_1:product:0lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/BiasAdd_2BiasAddlstm_cell_3/MatMul_2:product:0lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/BiasAdd_3BiasAddlstm_cell_3/MatMul_3:product:0lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬{
lstm_cell_3/mul_4Mulinitial_statelstm_cell_3/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬{
lstm_cell_3/mul_5Mulinitial_statelstm_cell_3/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬{
lstm_cell_3/mul_6Mulinitial_statelstm_cell_3/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬{
lstm_cell_3/mul_7Mulinitial_statelstm_cell_3/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/ReadVariableOpReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0p
lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  r
!lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell_3/strided_sliceStridedSlice"lstm_cell_3/ReadVariableOp:value:0(lstm_cell_3/strided_slice/stack:output:0*lstm_cell_3/strided_slice/stack_1:output:0*lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
lstm_cell_3/MatMul_4MatMullstm_cell_3/mul_4:z:0"lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/addAddV2lstm_cell_3/BiasAdd:output:0lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
lstm_cell_3/SigmoidSigmoidlstm_cell_3/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/ReadVariableOp_1ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0r
!lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  t
#lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  t
#lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
lstm_cell_3/strided_slice_1StridedSlice$lstm_cell_3/ReadVariableOp_1:value:0*lstm_cell_3/strided_slice_1/stack:output:0,lstm_cell_3/strided_slice_1/stack_1:output:0,lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
lstm_cell_3/MatMul_5MatMullstm_cell_3/mul_5:z:0$lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/add_1AddV2lstm_cell_3/BiasAdd_1:output:0lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬w
lstm_cell_3/mul_8Mullstm_cell_3/Sigmoid_1:y:0initial_state_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/ReadVariableOp_2ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0r
!lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  t
#lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
lstm_cell_3/strided_slice_2StridedSlice$lstm_cell_3/ReadVariableOp_2:value:0*lstm_cell_3/strided_slice_2/stack:output:0,lstm_cell_3/strided_slice_2/stack_1:output:0,lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
lstm_cell_3/MatMul_6MatMullstm_cell_3/mul_6:z:0$lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/add_2AddV2lstm_cell_3/BiasAdd_2:output:0lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬b
lstm_cell_3/TanhTanhlstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬z
lstm_cell_3/mul_9Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬{
lstm_cell_3/add_3AddV2lstm_cell_3/mul_8:z:0lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/ReadVariableOp_3ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0r
!lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
lstm_cell_3/strided_slice_3StridedSlice$lstm_cell_3/ReadVariableOp_3:value:0*lstm_cell_3/strided_slice_3/stack:output:0,lstm_cell_3/strided_slice_3/stack_1:output:0,lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
lstm_cell_3/MatMul_7MatMullstm_cell_3/mul_7:z:0$lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/add_4AddV2lstm_cell_3/BiasAdd_3:output:0lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬d
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/mul_10Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0initial_stateinitial_state_1strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_3_split_readvariableop_resource+lstm_cell_3_split_1_readvariableop_resource#lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_99024*
condR
while_cond_99023*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
NoOpNoOp^lstm_cell_3/ReadVariableOp^lstm_cell_3/ReadVariableOp_1^lstm_cell_3/ReadVariableOp_2^lstm_cell_3/ReadVariableOp_3!^lstm_cell_3/split/ReadVariableOp#^lstm_cell_3/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : 28
lstm_cell_3/ReadVariableOplstm_cell_3/ReadVariableOp2<
lstm_cell_3/ReadVariableOp_1lstm_cell_3/ReadVariableOp_12<
lstm_cell_3/ReadVariableOp_2lstm_cell_3/ReadVariableOp_22<
lstm_cell_3/ReadVariableOp_3lstm_cell_3/ReadVariableOp_32D
 lstm_cell_3/split/ReadVariableOp lstm_cell_3/split/ReadVariableOp2H
"lstm_cell_3/split_1/ReadVariableOp"lstm_cell_3/split_1/ReadVariableOp2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:WS
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
'
_user_specified_nameinitial_state:WS
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
'
_user_specified_nameinitial_state
á"
Ù
while_body_98065
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_3_98089_0:
È°	(
while_lstm_cell_3_98091_0:	°	-
while_lstm_cell_3_98093_0:
¬°	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_3_98089:
È°	&
while_lstm_cell_3_98091:	°	+
while_lstm_cell_3_98093:
¬°	¢)while/lstm_cell_3/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype0°
)while/lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_3_98089_0while_lstm_cell_3_98091_0while_lstm_cell_3_98093_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_98051Û
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_3/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity2while/lstm_cell_3/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/Identity_5Identity2while/lstm_cell_3/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬x

while/NoOpNoOp*^while/lstm_cell_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_3_98089while_lstm_cell_3_98089_0"4
while_lstm_cell_3_98091while_lstm_cell_3_98091_0"4
while_lstm_cell_3_98093while_lstm_cell_3_98093_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : 2V
)while/lstm_cell_3/StatefulPartitionedCall)while/lstm_cell_3/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
: 

Ç
K__inference_time_distributed_layer_call_and_return_conditional_losses_98496

inputs
dense_98486:
¬¸d
dense_98488:	¸d
identity¢dense/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ï
dense/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_98486dense_98488*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_98485\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :¸d
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape&dense/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸do
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸df
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
ïÂ
	
while_body_101220
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
1while_lstm_cell_3_split_readvariableop_resource_0:
È°	B
3while_lstm_cell_3_split_1_readvariableop_resource_0:	°	?
+while_lstm_cell_3_readvariableop_resource_0:
¬°	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
/while_lstm_cell_3_split_readvariableop_resource:
È°	@
1while_lstm_cell_3_split_1_readvariableop_resource:	°	=
)while_lstm_cell_3_readvariableop_resource:
¬°	¢ while/lstm_cell_3/ReadVariableOp¢"while/lstm_cell_3/ReadVariableOp_1¢"while/lstm_cell_3/ReadVariableOp_2¢"while/lstm_cell_3/ReadVariableOp_3¢&while/lstm_cell_3/split/ReadVariableOp¢(while/lstm_cell_3/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype0
!while/lstm_cell_3/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?®
while/lstm_cell_3/ones_likeFill*while/lstm_cell_3/ones_like/Shape:output:0*while/lstm_cell_3/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈd
while/lstm_cell_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?§
while/lstm_cell_3/dropout/MulMul$while/lstm_cell_3/ones_like:output:0(while/lstm_cell_3/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈs
while/lstm_cell_3/dropout/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:±
6while/lstm_cell_3/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_3/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0m
(while/lstm_cell_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>Ý
&while/lstm_cell_3/dropout/GreaterEqualGreaterEqual?while/lstm_cell_3/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
while/lstm_cell_3/dropout/CastCast*while/lstm_cell_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
while/lstm_cell_3/dropout/Mul_1Mul!while/lstm_cell_3/dropout/Mul:z:0"while/lstm_cell_3/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈf
!while/lstm_cell_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?«
while/lstm_cell_3/dropout_1/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈu
!while/lstm_cell_3/dropout_1/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:µ
8while/lstm_cell_3/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0o
*while/lstm_cell_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>ã
(while/lstm_cell_3/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 while/lstm_cell_3/dropout_1/CastCast,while/lstm_cell_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¦
!while/lstm_cell_3/dropout_1/Mul_1Mul#while/lstm_cell_3/dropout_1/Mul:z:0$while/lstm_cell_3/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈf
!while/lstm_cell_3/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?«
while/lstm_cell_3/dropout_2/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈu
!while/lstm_cell_3/dropout_2/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:µ
8while/lstm_cell_3/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0o
*while/lstm_cell_3/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>ã
(while/lstm_cell_3/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 while/lstm_cell_3/dropout_2/CastCast,while/lstm_cell_3/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¦
!while/lstm_cell_3/dropout_2/Mul_1Mul#while/lstm_cell_3/dropout_2/Mul:z:0$while/lstm_cell_3/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈf
!while/lstm_cell_3/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?«
while/lstm_cell_3/dropout_3/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈu
!while/lstm_cell_3/dropout_3/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:µ
8while/lstm_cell_3/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0o
*while/lstm_cell_3/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>ã
(while/lstm_cell_3/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 while/lstm_cell_3/dropout_3/CastCast,while/lstm_cell_3/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¦
!while/lstm_cell_3/dropout_3/Mul_1Mul#while/lstm_cell_3/dropout_3/Mul:z:0$while/lstm_cell_3/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈf
#while/lstm_cell_3/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?´
while/lstm_cell_3/ones_like_1Fill,while/lstm_cell_3/ones_like_1/Shape:output:0,while/lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
!while/lstm_cell_3/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?­
while/lstm_cell_3/dropout_4/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_4/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬w
!while/lstm_cell_3/dropout_4/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:µ
8while/lstm_cell_3/dropout_4/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_4/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0o
*while/lstm_cell_3/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>ã
(while/lstm_cell_3/dropout_4/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_4/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 while/lstm_cell_3/dropout_4/CastCast,while/lstm_cell_3/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
!while/lstm_cell_3/dropout_4/Mul_1Mul#while/lstm_cell_3/dropout_4/Mul:z:0$while/lstm_cell_3/dropout_4/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
!while/lstm_cell_3/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?­
while/lstm_cell_3/dropout_5/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬w
!while/lstm_cell_3/dropout_5/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:µ
8while/lstm_cell_3/dropout_5/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_5/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0o
*while/lstm_cell_3/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>ã
(while/lstm_cell_3/dropout_5/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_5/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 while/lstm_cell_3/dropout_5/CastCast,while/lstm_cell_3/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
!while/lstm_cell_3/dropout_5/Mul_1Mul#while/lstm_cell_3/dropout_5/Mul:z:0$while/lstm_cell_3/dropout_5/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
!while/lstm_cell_3/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?­
while/lstm_cell_3/dropout_6/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_6/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬w
!while/lstm_cell_3/dropout_6/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:µ
8while/lstm_cell_3/dropout_6/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_6/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0o
*while/lstm_cell_3/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>ã
(while/lstm_cell_3/dropout_6/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_6/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 while/lstm_cell_3/dropout_6/CastCast,while/lstm_cell_3/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
!while/lstm_cell_3/dropout_6/Mul_1Mul#while/lstm_cell_3/dropout_6/Mul:z:0$while/lstm_cell_3/dropout_6/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
!while/lstm_cell_3/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?­
while/lstm_cell_3/dropout_7/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_7/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬w
!while/lstm_cell_3/dropout_7/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:µ
8while/lstm_cell_3/dropout_7/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_7/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0o
*while/lstm_cell_3/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>ã
(while/lstm_cell_3/dropout_7/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_7/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 while/lstm_cell_3/dropout_7/CastCast,while/lstm_cell_3/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
!while/lstm_cell_3/dropout_7/Mul_1Mul#while/lstm_cell_3/dropout_7/Mul:z:0$while/lstm_cell_3/dropout_7/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
while/lstm_cell_3/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell_3/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈª
while/lstm_cell_3/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈª
while/lstm_cell_3/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈª
while/lstm_cell_3/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈc
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&while/lstm_cell_3/split/ReadVariableOpReadVariableOp1while_lstm_cell_3_split_readvariableop_resource_0* 
_output_shapes
:
È°	*
dtype0Ü
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0.while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È¬:
È¬:
È¬:
È¬*
	num_split
while/lstm_cell_3/MatMulMatMulwhile/lstm_cell_3/mul:z:0 while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/MatMul_1MatMulwhile/lstm_cell_3/mul_1:z:0 while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/MatMul_2MatMulwhile/lstm_cell_3/mul_2:z:0 while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/MatMul_3MatMulwhile/lstm_cell_3/mul_3:z:0 while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
#while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:°	*
dtype0Î
while/lstm_cell_3/split_1Split,while/lstm_cell_3/split_1/split_dim:output:00while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:¬:¬:¬:¬*
	num_split
while/lstm_cell_3/BiasAddBiasAdd"while/lstm_cell_3/MatMul:product:0"while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/lstm_cell_3/BiasAdd_1BiasAdd$while/lstm_cell_3/MatMul_1:product:0"while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/lstm_cell_3/BiasAdd_2BiasAdd$while/lstm_cell_3/MatMul_2:product:0"while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/lstm_cell_3/BiasAdd_3BiasAdd$while/lstm_cell_3/MatMul_3:product:0"while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_4Mulwhile_placeholder_2%while/lstm_cell_3/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_5Mulwhile_placeholder_2%while/lstm_cell_3/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_6Mulwhile_placeholder_2%while/lstm_cell_3/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_7Mulwhile_placeholder_2%while/lstm_cell_3/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 while/lstm_cell_3/ReadVariableOpReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0v
%while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  x
'while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell_3/strided_sliceStridedSlice(while/lstm_cell_3/ReadVariableOp:value:0.while/lstm_cell_3/strided_slice/stack:output:00while/lstm_cell_3/strided_slice/stack_1:output:00while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_4MatMulwhile/lstm_cell_3/mul_4:z:0(while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/addAddV2"while/lstm_cell_3/BiasAdd:output:0$while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r
while/lstm_cell_3/SigmoidSigmoidwhile/lstm_cell_3/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"while/lstm_cell_3/ReadVariableOp_1ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0x
'while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  z
)while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  z
)while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ñ
!while/lstm_cell_3/strided_slice_1StridedSlice*while/lstm_cell_3/ReadVariableOp_1:value:00while/lstm_cell_3/strided_slice_1/stack:output:02while/lstm_cell_3/strided_slice_1/stack_1:output:02while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask 
while/lstm_cell_3/MatMul_5MatMulwhile/lstm_cell_3/mul_5:z:0*while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/add_1AddV2$while/lstm_cell_3/BiasAdd_1:output:0$while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬v
while/lstm_cell_3/Sigmoid_1Sigmoidwhile/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_8Mulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"while/lstm_cell_3/ReadVariableOp_2ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0x
'while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  z
)while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ñ
!while/lstm_cell_3/strided_slice_2StridedSlice*while/lstm_cell_3/ReadVariableOp_2:value:00while/lstm_cell_3/strided_slice_2/stack:output:02while/lstm_cell_3/strided_slice_2/stack_1:output:02while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask 
while/lstm_cell_3/MatMul_6MatMulwhile/lstm_cell_3/mul_6:z:0*while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/add_2AddV2$while/lstm_cell_3/BiasAdd_2:output:0$while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬n
while/lstm_cell_3/TanhTanhwhile/lstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_9Mulwhile/lstm_cell_3/Sigmoid:y:0while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/add_3AddV2while/lstm_cell_3/mul_8:z:0while/lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"while/lstm_cell_3/ReadVariableOp_3ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0x
'while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ñ
!while/lstm_cell_3/strided_slice_3StridedSlice*while/lstm_cell_3/ReadVariableOp_3:value:00while/lstm_cell_3/strided_slice_3/stack:output:02while/lstm_cell_3/strided_slice_3/stack_1:output:02while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask 
while/lstm_cell_3/MatMul_7MatMulwhile/lstm_cell_3/mul_7:z:0*while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/add_4AddV2$while/lstm_cell_3/BiasAdd_3:output:0$while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬v
while/lstm_cell_3/Sigmoid_2Sigmoidwhile/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬p
while/lstm_cell_3/Tanh_1Tanhwhile/lstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_10Mulwhile/lstm_cell_3/Sigmoid_2:y:0while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_3/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬y
while/Identity_5Identitywhile/lstm_cell_3/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬²

while/NoOpNoOp!^while/lstm_cell_3/ReadVariableOp#^while/lstm_cell_3/ReadVariableOp_1#^while/lstm_cell_3/ReadVariableOp_2#^while/lstm_cell_3/ReadVariableOp_3'^while/lstm_cell_3/split/ReadVariableOp)^while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_3_readvariableop_resource+while_lstm_cell_3_readvariableop_resource_0"h
1while_lstm_cell_3_split_1_readvariableop_resource3while_lstm_cell_3_split_1_readvariableop_resource_0"d
/while_lstm_cell_3_split_readvariableop_resource1while_lstm_cell_3_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : 2D
 while/lstm_cell_3/ReadVariableOp while/lstm_cell_3/ReadVariableOp2H
"while/lstm_cell_3/ReadVariableOp_1"while/lstm_cell_3/ReadVariableOp_12H
"while/lstm_cell_3/ReadVariableOp_2"while/lstm_cell_3/ReadVariableOp_22H
"while/lstm_cell_3/ReadVariableOp_3"while/lstm_cell_3/ReadVariableOp_32P
&while/lstm_cell_3/split/ReadVariableOp&while/lstm_cell_3/split/ReadVariableOp2T
(while/lstm_cell_3/split_1/ReadVariableOp(while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
: 
é	
¦
G__inference_embedding_1_layer_call_and_return_conditional_losses_100142

inputs+
embedding_lookup_100136:
¸dÈ
identity¢embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ
embedding_lookupResourceGatherembedding_lookup_100136Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/100136*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*
dtype0¬
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/100136*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
îÂ
	
while_body_99024
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
1while_lstm_cell_3_split_readvariableop_resource_0:
È°	B
3while_lstm_cell_3_split_1_readvariableop_resource_0:	°	?
+while_lstm_cell_3_readvariableop_resource_0:
¬°	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
/while_lstm_cell_3_split_readvariableop_resource:
È°	@
1while_lstm_cell_3_split_1_readvariableop_resource:	°	=
)while_lstm_cell_3_readvariableop_resource:
¬°	¢ while/lstm_cell_3/ReadVariableOp¢"while/lstm_cell_3/ReadVariableOp_1¢"while/lstm_cell_3/ReadVariableOp_2¢"while/lstm_cell_3/ReadVariableOp_3¢&while/lstm_cell_3/split/ReadVariableOp¢(while/lstm_cell_3/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype0
!while/lstm_cell_3/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?®
while/lstm_cell_3/ones_likeFill*while/lstm_cell_3/ones_like/Shape:output:0*while/lstm_cell_3/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈd
while/lstm_cell_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?§
while/lstm_cell_3/dropout/MulMul$while/lstm_cell_3/ones_like:output:0(while/lstm_cell_3/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈs
while/lstm_cell_3/dropout/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:±
6while/lstm_cell_3/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_3/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0m
(while/lstm_cell_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>Ý
&while/lstm_cell_3/dropout/GreaterEqualGreaterEqual?while/lstm_cell_3/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
while/lstm_cell_3/dropout/CastCast*while/lstm_cell_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
while/lstm_cell_3/dropout/Mul_1Mul!while/lstm_cell_3/dropout/Mul:z:0"while/lstm_cell_3/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈf
!while/lstm_cell_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?«
while/lstm_cell_3/dropout_1/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈu
!while/lstm_cell_3/dropout_1/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:µ
8while/lstm_cell_3/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0o
*while/lstm_cell_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>ã
(while/lstm_cell_3/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 while/lstm_cell_3/dropout_1/CastCast,while/lstm_cell_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¦
!while/lstm_cell_3/dropout_1/Mul_1Mul#while/lstm_cell_3/dropout_1/Mul:z:0$while/lstm_cell_3/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈf
!while/lstm_cell_3/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?«
while/lstm_cell_3/dropout_2/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈu
!while/lstm_cell_3/dropout_2/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:µ
8while/lstm_cell_3/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0o
*while/lstm_cell_3/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>ã
(while/lstm_cell_3/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 while/lstm_cell_3/dropout_2/CastCast,while/lstm_cell_3/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¦
!while/lstm_cell_3/dropout_2/Mul_1Mul#while/lstm_cell_3/dropout_2/Mul:z:0$while/lstm_cell_3/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈf
!while/lstm_cell_3/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?«
while/lstm_cell_3/dropout_3/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈu
!while/lstm_cell_3/dropout_3/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:µ
8while/lstm_cell_3/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0o
*while/lstm_cell_3/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>ã
(while/lstm_cell_3/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 while/lstm_cell_3/dropout_3/CastCast,while/lstm_cell_3/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¦
!while/lstm_cell_3/dropout_3/Mul_1Mul#while/lstm_cell_3/dropout_3/Mul:z:0$while/lstm_cell_3/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈf
#while/lstm_cell_3/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?´
while/lstm_cell_3/ones_like_1Fill,while/lstm_cell_3/ones_like_1/Shape:output:0,while/lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
!while/lstm_cell_3/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?­
while/lstm_cell_3/dropout_4/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_4/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬w
!while/lstm_cell_3/dropout_4/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:µ
8while/lstm_cell_3/dropout_4/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_4/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0o
*while/lstm_cell_3/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>ã
(while/lstm_cell_3/dropout_4/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_4/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 while/lstm_cell_3/dropout_4/CastCast,while/lstm_cell_3/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
!while/lstm_cell_3/dropout_4/Mul_1Mul#while/lstm_cell_3/dropout_4/Mul:z:0$while/lstm_cell_3/dropout_4/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
!while/lstm_cell_3/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?­
while/lstm_cell_3/dropout_5/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬w
!while/lstm_cell_3/dropout_5/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:µ
8while/lstm_cell_3/dropout_5/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_5/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0o
*while/lstm_cell_3/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>ã
(while/lstm_cell_3/dropout_5/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_5/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 while/lstm_cell_3/dropout_5/CastCast,while/lstm_cell_3/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
!while/lstm_cell_3/dropout_5/Mul_1Mul#while/lstm_cell_3/dropout_5/Mul:z:0$while/lstm_cell_3/dropout_5/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
!while/lstm_cell_3/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?­
while/lstm_cell_3/dropout_6/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_6/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬w
!while/lstm_cell_3/dropout_6/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:µ
8while/lstm_cell_3/dropout_6/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_6/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0o
*while/lstm_cell_3/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>ã
(while/lstm_cell_3/dropout_6/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_6/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 while/lstm_cell_3/dropout_6/CastCast,while/lstm_cell_3/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
!while/lstm_cell_3/dropout_6/Mul_1Mul#while/lstm_cell_3/dropout_6/Mul:z:0$while/lstm_cell_3/dropout_6/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
!while/lstm_cell_3/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?­
while/lstm_cell_3/dropout_7/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_7/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬w
!while/lstm_cell_3/dropout_7/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:µ
8while/lstm_cell_3/dropout_7/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_7/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0o
*while/lstm_cell_3/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>ã
(while/lstm_cell_3/dropout_7/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_7/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 while/lstm_cell_3/dropout_7/CastCast,while/lstm_cell_3/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
!while/lstm_cell_3/dropout_7/Mul_1Mul#while/lstm_cell_3/dropout_7/Mul:z:0$while/lstm_cell_3/dropout_7/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
while/lstm_cell_3/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell_3/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈª
while/lstm_cell_3/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈª
while/lstm_cell_3/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈª
while/lstm_cell_3/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈc
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&while/lstm_cell_3/split/ReadVariableOpReadVariableOp1while_lstm_cell_3_split_readvariableop_resource_0* 
_output_shapes
:
È°	*
dtype0Ü
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0.while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È¬:
È¬:
È¬:
È¬*
	num_split
while/lstm_cell_3/MatMulMatMulwhile/lstm_cell_3/mul:z:0 while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/MatMul_1MatMulwhile/lstm_cell_3/mul_1:z:0 while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/MatMul_2MatMulwhile/lstm_cell_3/mul_2:z:0 while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/MatMul_3MatMulwhile/lstm_cell_3/mul_3:z:0 while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
#while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:°	*
dtype0Î
while/lstm_cell_3/split_1Split,while/lstm_cell_3/split_1/split_dim:output:00while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:¬:¬:¬:¬*
	num_split
while/lstm_cell_3/BiasAddBiasAdd"while/lstm_cell_3/MatMul:product:0"while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/lstm_cell_3/BiasAdd_1BiasAdd$while/lstm_cell_3/MatMul_1:product:0"while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/lstm_cell_3/BiasAdd_2BiasAdd$while/lstm_cell_3/MatMul_2:product:0"while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/lstm_cell_3/BiasAdd_3BiasAdd$while/lstm_cell_3/MatMul_3:product:0"while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_4Mulwhile_placeholder_2%while/lstm_cell_3/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_5Mulwhile_placeholder_2%while/lstm_cell_3/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_6Mulwhile_placeholder_2%while/lstm_cell_3/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_7Mulwhile_placeholder_2%while/lstm_cell_3/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 while/lstm_cell_3/ReadVariableOpReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0v
%while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  x
'while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell_3/strided_sliceStridedSlice(while/lstm_cell_3/ReadVariableOp:value:0.while/lstm_cell_3/strided_slice/stack:output:00while/lstm_cell_3/strided_slice/stack_1:output:00while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_4MatMulwhile/lstm_cell_3/mul_4:z:0(while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/addAddV2"while/lstm_cell_3/BiasAdd:output:0$while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r
while/lstm_cell_3/SigmoidSigmoidwhile/lstm_cell_3/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"while/lstm_cell_3/ReadVariableOp_1ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0x
'while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  z
)while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  z
)while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ñ
!while/lstm_cell_3/strided_slice_1StridedSlice*while/lstm_cell_3/ReadVariableOp_1:value:00while/lstm_cell_3/strided_slice_1/stack:output:02while/lstm_cell_3/strided_slice_1/stack_1:output:02while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask 
while/lstm_cell_3/MatMul_5MatMulwhile/lstm_cell_3/mul_5:z:0*while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/add_1AddV2$while/lstm_cell_3/BiasAdd_1:output:0$while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬v
while/lstm_cell_3/Sigmoid_1Sigmoidwhile/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_8Mulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"while/lstm_cell_3/ReadVariableOp_2ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0x
'while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  z
)while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ñ
!while/lstm_cell_3/strided_slice_2StridedSlice*while/lstm_cell_3/ReadVariableOp_2:value:00while/lstm_cell_3/strided_slice_2/stack:output:02while/lstm_cell_3/strided_slice_2/stack_1:output:02while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask 
while/lstm_cell_3/MatMul_6MatMulwhile/lstm_cell_3/mul_6:z:0*while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/add_2AddV2$while/lstm_cell_3/BiasAdd_2:output:0$while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬n
while/lstm_cell_3/TanhTanhwhile/lstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_9Mulwhile/lstm_cell_3/Sigmoid:y:0while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/add_3AddV2while/lstm_cell_3/mul_8:z:0while/lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"while/lstm_cell_3/ReadVariableOp_3ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0x
'while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      z
)while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ñ
!while/lstm_cell_3/strided_slice_3StridedSlice*while/lstm_cell_3/ReadVariableOp_3:value:00while/lstm_cell_3/strided_slice_3/stack:output:02while/lstm_cell_3/strided_slice_3/stack_1:output:02while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask 
while/lstm_cell_3/MatMul_7MatMulwhile/lstm_cell_3/mul_7:z:0*while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/add_4AddV2$while/lstm_cell_3/BiasAdd_3:output:0$while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬v
while/lstm_cell_3/Sigmoid_2Sigmoidwhile/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬p
while/lstm_cell_3/Tanh_1Tanhwhile/lstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_3/mul_10Mulwhile/lstm_cell_3/Sigmoid_2:y:0while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_3/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬y
while/Identity_5Identitywhile/lstm_cell_3/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬²

while/NoOpNoOp!^while/lstm_cell_3/ReadVariableOp#^while/lstm_cell_3/ReadVariableOp_1#^while/lstm_cell_3/ReadVariableOp_2#^while/lstm_cell_3/ReadVariableOp_3'^while/lstm_cell_3/split/ReadVariableOp)^while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_3_readvariableop_resource+while_lstm_cell_3_readvariableop_resource_0"h
1while_lstm_cell_3_split_1_readvariableop_resource3while_lstm_cell_3_split_1_readvariableop_resource_0"d
/while_lstm_cell_3_split_readvariableop_resource1while_lstm_cell_3_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : 2D
 while/lstm_cell_3/ReadVariableOp while/lstm_cell_3/ReadVariableOp2H
"while/lstm_cell_3/ReadVariableOp_1"while/lstm_cell_3/ReadVariableOp_12H
"while/lstm_cell_3/ReadVariableOp_2"while/lstm_cell_3/ReadVariableOp_22H
"while/lstm_cell_3/ReadVariableOp_3"while/lstm_cell_3/ReadVariableOp_32P
&while/lstm_cell_3/split/ReadVariableOp&while/lstm_cell_3/split/ReadVariableOp2T
(while/lstm_cell_3/split_1/ReadVariableOp(while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
: 

©
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_98317

inputs

states
states_11
split_readvariableop_resource:
È°	.
split_1_readvariableop_resource:	°	+
readvariableop_resource:
¬°	
identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?q
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈO
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈT
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?u
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈQ
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>­
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈt
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈp
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈT
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?u
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈQ
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>­
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈt
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈp
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈT
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?u
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈQ
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>­
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈt
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈp
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈG
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬T
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬S
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0]
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>­
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬t
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬p
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬T
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬S
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0]
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>­
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬t
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬p
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬T
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬S
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0]
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>­
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬t
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬p
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬T
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬S
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0]
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>­
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬t
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬p
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬X
mulMulinputsdropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ\
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ\
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ\
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
È°	*
dtype0¦
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È¬:
È¬:
È¬:
È¬*
	num_split\
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:°	*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:¬:¬:¬:¬*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬\
mul_4Mulstatesdropout_4/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬\
mul_5Mulstatesdropout_5/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬\
mul_6Mulstatesdropout_6/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬\
mul_7Mulstatesdropout_7/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
¬°	*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskh
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
¬°	*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskj
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬X
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
¬°	*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskj
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬V
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬W
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
¬°	*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskj
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬R
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬L
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬[
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
IdentityIdentity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬\

Identity_1Identity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬[

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬À
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_namestates
Í

'__inference_lstm_3_layer_call_fn_100206

inputs
initial_state_0
initial_state_1
unknown:
È°	
	unknown_0:	°	
	unknown_1:
¬°	
identity

identity_1

identity_2¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputsinitial_state_0initial_state_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_99224}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:YU
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
)
_user_specified_nameinitial_state/0:YU
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
)
_user_specified_nameinitial_state/1

¡
1__inference_time_distributed_layer_call_fn_101429

inputs
unknown:
¬¸d
	unknown_0:	¸d
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_98496}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
Ï

B__inference_lstm_3_layer_call_and_return_conditional_losses_100451
inputs_0=
)lstm_cell_3_split_readvariableop_resource:
È°	:
+lstm_cell_3_split_1_readvariableop_resource:	°	7
#lstm_cell_3_readvariableop_resource:
¬°	
identity

identity_1

identity_2¢lstm_cell_3/ReadVariableOp¢lstm_cell_3/ReadVariableOp_1¢lstm_cell_3/ReadVariableOp_2¢lstm_cell_3/ReadVariableOp_3¢ lstm_cell_3/split/ReadVariableOp¢"lstm_cell_3/split_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¬s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¬w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_maskc
lstm_cell_3/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_3/ones_likeFill$lstm_cell_3/ones_like/Shape:output:0$lstm_cell_3/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ[
lstm_cell_3/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¢
lstm_cell_3/ones_like_1Fill&lstm_cell_3/ones_like_1/Shape:output:0&lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/mulMulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/mul_1Mulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/mul_2Mulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/mul_3Mulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ]
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_cell_3/split/ReadVariableOpReadVariableOp)lstm_cell_3_split_readvariableop_resource* 
_output_shapes
:
È°	*
dtype0Ê
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0(lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È¬:
È¬:
È¬:
È¬*
	num_split
lstm_cell_3/MatMulMatMullstm_cell_3/mul:z:0lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/MatMul_1MatMullstm_cell_3/mul_1:z:0lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/MatMul_2MatMullstm_cell_3/mul_2:z:0lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/MatMul_3MatMullstm_cell_3/mul_3:z:0lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬_
lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"lstm_cell_3/split_1/ReadVariableOpReadVariableOp+lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:°	*
dtype0¼
lstm_cell_3/split_1Split&lstm_cell_3/split_1/split_dim:output:0*lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:¬:¬:¬:¬*
	num_split
lstm_cell_3/BiasAddBiasAddlstm_cell_3/MatMul:product:0lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/BiasAdd_1BiasAddlstm_cell_3/MatMul_1:product:0lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/BiasAdd_2BiasAddlstm_cell_3/MatMul_2:product:0lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/BiasAdd_3BiasAddlstm_cell_3/MatMul_3:product:0lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬}
lstm_cell_3/mul_4Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬}
lstm_cell_3/mul_5Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬}
lstm_cell_3/mul_6Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬}
lstm_cell_3/mul_7Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/ReadVariableOpReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0p
lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  r
!lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell_3/strided_sliceStridedSlice"lstm_cell_3/ReadVariableOp:value:0(lstm_cell_3/strided_slice/stack:output:0*lstm_cell_3/strided_slice/stack_1:output:0*lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
lstm_cell_3/MatMul_4MatMullstm_cell_3/mul_4:z:0"lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/addAddV2lstm_cell_3/BiasAdd:output:0lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
lstm_cell_3/SigmoidSigmoidlstm_cell_3/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/ReadVariableOp_1ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0r
!lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  t
#lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  t
#lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
lstm_cell_3/strided_slice_1StridedSlice$lstm_cell_3/ReadVariableOp_1:value:0*lstm_cell_3/strided_slice_1/stack:output:0,lstm_cell_3/strided_slice_1/stack_1:output:0,lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
lstm_cell_3/MatMul_5MatMullstm_cell_3/mul_5:z:0$lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/add_1AddV2lstm_cell_3/BiasAdd_1:output:0lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬x
lstm_cell_3/mul_8Mullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/ReadVariableOp_2ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0r
!lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  t
#lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
lstm_cell_3/strided_slice_2StridedSlice$lstm_cell_3/ReadVariableOp_2:value:0*lstm_cell_3/strided_slice_2/stack:output:0,lstm_cell_3/strided_slice_2/stack_1:output:0,lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
lstm_cell_3/MatMul_6MatMullstm_cell_3/mul_6:z:0$lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/add_2AddV2lstm_cell_3/BiasAdd_2:output:0lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬b
lstm_cell_3/TanhTanhlstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬z
lstm_cell_3/mul_9Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬{
lstm_cell_3/add_3AddV2lstm_cell_3/mul_8:z:0lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/ReadVariableOp_3ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0r
!lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
lstm_cell_3/strided_slice_3StridedSlice$lstm_cell_3/ReadVariableOp_3:value:0*lstm_cell_3/strided_slice_3/stack:output:0,lstm_cell_3/strided_slice_3/stack_1:output:0,lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
lstm_cell_3/MatMul_7MatMullstm_cell_3/mul_7:z:0$lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/add_4AddV2lstm_cell_3/BiasAdd_3:output:0lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬d
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/mul_10Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ù
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_3_split_readvariableop_resource+lstm_cell_3_split_1_readvariableop_resource#lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_100315*
condR
while_cond_100314*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
NoOpNoOp^lstm_cell_3/ReadVariableOp^lstm_cell_3/ReadVariableOp_1^lstm_cell_3/ReadVariableOp_2^lstm_cell_3/ReadVariableOp_3!^lstm_cell_3/split/ReadVariableOp#^lstm_cell_3/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ: : : 28
lstm_cell_3/ReadVariableOplstm_cell_3/ReadVariableOp2<
lstm_cell_3/ReadVariableOp_1lstm_cell_3/ReadVariableOp_12<
lstm_cell_3/ReadVariableOp_2lstm_cell_3/ReadVariableOp_22<
lstm_cell_3/ReadVariableOp_3lstm_cell_3/ReadVariableOp_32D
 lstm_cell_3/split/ReadVariableOp lstm_cell_3/split/ReadVariableOp2H
"lstm_cell_3/split_1/ReadVariableOp"lstm_cell_3/split_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
inputs/0
°
Ã
"__inference__traced_restore_101838
file_prefix;
'assignvariableop_embedding_1_embeddings:
¸dÈ@
,assignvariableop_1_lstm_3_lstm_cell_3_kernel:
È°	J
6assignvariableop_2_lstm_3_lstm_cell_3_recurrent_kernel:
¬°	9
*assignvariableop_3_lstm_3_lstm_cell_3_bias:	°	>
*assignvariableop_4_time_distributed_kernel:
¬¸d7
(assignvariableop_5_time_distributed_bias:	¸d

identity_7¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*·
value­BªB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH~
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B Á
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp'assignvariableop_embedding_1_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp,assignvariableop_1_lstm_3_lstm_cell_3_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_2AssignVariableOp6assignvariableop_2_lstm_3_lstm_cell_3_recurrent_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp*assignvariableop_3_lstm_3_lstm_cell_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp*assignvariableop_4_time_distributed_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp(assignvariableop_5_time_distributed_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ö

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_7IdentityIdentity_6:output:0^NoOp_1*
T0*
_output_shapes
: Ä
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 "!

identity_7Identity_7:output:0*!
_input_shapes
: : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ÇÙ

lstm_3_while_body_99879*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3'
#lstm_3_while_lstm_3_strided_slice_0e
alstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0L
8lstm_3_while_lstm_cell_3_split_readvariableop_resource_0:
È°	I
:lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0:	°	F
2lstm_3_while_lstm_cell_3_readvariableop_resource_0:
¬°	
lstm_3_while_identity
lstm_3_while_identity_1
lstm_3_while_identity_2
lstm_3_while_identity_3
lstm_3_while_identity_4
lstm_3_while_identity_5%
!lstm_3_while_lstm_3_strided_slicec
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensorJ
6lstm_3_while_lstm_cell_3_split_readvariableop_resource:
È°	G
8lstm_3_while_lstm_cell_3_split_1_readvariableop_resource:	°	D
0lstm_3_while_lstm_cell_3_readvariableop_resource:
¬°	¢'lstm_3/while/lstm_cell_3/ReadVariableOp¢)lstm_3/while/lstm_cell_3/ReadVariableOp_1¢)lstm_3/while/lstm_cell_3/ReadVariableOp_2¢)lstm_3/while/lstm_cell_3/ReadVariableOp_3¢-lstm_3/while/lstm_cell_3/split/ReadVariableOp¢/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   Ê
0lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0lstm_3_while_placeholderGlstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype0
(lstm_3/while/lstm_cell_3/ones_like/ShapeShape7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:m
(lstm_3/while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ã
"lstm_3/while/lstm_cell_3/ones_likeFill1lstm_3/while/lstm_cell_3/ones_like/Shape:output:01lstm_3/while/lstm_cell_3/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈk
&lstm_3/while/lstm_cell_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?¼
$lstm_3/while/lstm_cell_3/dropout/MulMul+lstm_3/while/lstm_cell_3/ones_like:output:0/lstm_3/while/lstm_cell_3/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
&lstm_3/while/lstm_cell_3/dropout/ShapeShape+lstm_3/while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:¿
=lstm_3/while/lstm_cell_3/dropout/random_uniform/RandomUniformRandomUniform/lstm_3/while/lstm_cell_3/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0t
/lstm_3/while/lstm_cell_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>ò
-lstm_3/while/lstm_cell_3/dropout/GreaterEqualGreaterEqualFlstm_3/while/lstm_cell_3/dropout/random_uniform/RandomUniform:output:08lstm_3/while/lstm_cell_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¢
%lstm_3/while/lstm_cell_3/dropout/CastCast1lstm_3/while/lstm_cell_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈµ
&lstm_3/while/lstm_cell_3/dropout/Mul_1Mul(lstm_3/while/lstm_cell_3/dropout/Mul:z:0)lstm_3/while/lstm_cell_3/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈm
(lstm_3/while/lstm_cell_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?À
&lstm_3/while/lstm_cell_3/dropout_1/MulMul+lstm_3/while/lstm_cell_3/ones_like:output:01lstm_3/while/lstm_cell_3/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
(lstm_3/while/lstm_cell_3/dropout_1/ShapeShape+lstm_3/while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:Ã
?lstm_3/while/lstm_cell_3/dropout_1/random_uniform/RandomUniformRandomUniform1lstm_3/while/lstm_cell_3/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0v
1lstm_3/while/lstm_cell_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>ø
/lstm_3/while/lstm_cell_3/dropout_1/GreaterEqualGreaterEqualHlstm_3/while/lstm_cell_3/dropout_1/random_uniform/RandomUniform:output:0:lstm_3/while/lstm_cell_3/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¦
'lstm_3/while/lstm_cell_3/dropout_1/CastCast3lstm_3/while/lstm_cell_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ»
(lstm_3/while/lstm_cell_3/dropout_1/Mul_1Mul*lstm_3/while/lstm_cell_3/dropout_1/Mul:z:0+lstm_3/while/lstm_cell_3/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈm
(lstm_3/while/lstm_cell_3/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?À
&lstm_3/while/lstm_cell_3/dropout_2/MulMul+lstm_3/while/lstm_cell_3/ones_like:output:01lstm_3/while/lstm_cell_3/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
(lstm_3/while/lstm_cell_3/dropout_2/ShapeShape+lstm_3/while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:Ã
?lstm_3/while/lstm_cell_3/dropout_2/random_uniform/RandomUniformRandomUniform1lstm_3/while/lstm_cell_3/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0v
1lstm_3/while/lstm_cell_3/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>ø
/lstm_3/while/lstm_cell_3/dropout_2/GreaterEqualGreaterEqualHlstm_3/while/lstm_cell_3/dropout_2/random_uniform/RandomUniform:output:0:lstm_3/while/lstm_cell_3/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¦
'lstm_3/while/lstm_cell_3/dropout_2/CastCast3lstm_3/while/lstm_cell_3/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ»
(lstm_3/while/lstm_cell_3/dropout_2/Mul_1Mul*lstm_3/while/lstm_cell_3/dropout_2/Mul:z:0+lstm_3/while/lstm_cell_3/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈm
(lstm_3/while/lstm_cell_3/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?À
&lstm_3/while/lstm_cell_3/dropout_3/MulMul+lstm_3/while/lstm_cell_3/ones_like:output:01lstm_3/while/lstm_cell_3/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
(lstm_3/while/lstm_cell_3/dropout_3/ShapeShape+lstm_3/while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:Ã
?lstm_3/while/lstm_cell_3/dropout_3/random_uniform/RandomUniformRandomUniform1lstm_3/while/lstm_cell_3/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0v
1lstm_3/while/lstm_cell_3/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>ø
/lstm_3/while/lstm_cell_3/dropout_3/GreaterEqualGreaterEqualHlstm_3/while/lstm_cell_3/dropout_3/random_uniform/RandomUniform:output:0:lstm_3/while/lstm_cell_3/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¦
'lstm_3/while/lstm_cell_3/dropout_3/CastCast3lstm_3/while/lstm_cell_3/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ»
(lstm_3/while/lstm_cell_3/dropout_3/Mul_1Mul*lstm_3/while/lstm_cell_3/dropout_3/Mul:z:0+lstm_3/while/lstm_cell_3/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈt
*lstm_3/while/lstm_cell_3/ones_like_1/ShapeShapelstm_3_while_placeholder_2*
T0*
_output_shapes
:o
*lstm_3/while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?É
$lstm_3/while/lstm_cell_3/ones_like_1Fill3lstm_3/while/lstm_cell_3/ones_like_1/Shape:output:03lstm_3/while/lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
(lstm_3/while/lstm_cell_3/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Â
&lstm_3/while/lstm_cell_3/dropout_4/MulMul-lstm_3/while/lstm_cell_3/ones_like_1:output:01lstm_3/while/lstm_cell_3/dropout_4/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
(lstm_3/while/lstm_cell_3/dropout_4/ShapeShape-lstm_3/while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:Ã
?lstm_3/while/lstm_cell_3/dropout_4/random_uniform/RandomUniformRandomUniform1lstm_3/while/lstm_cell_3/dropout_4/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0v
1lstm_3/while/lstm_cell_3/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>ø
/lstm_3/while/lstm_cell_3/dropout_4/GreaterEqualGreaterEqualHlstm_3/while/lstm_cell_3/dropout_4/random_uniform/RandomUniform:output:0:lstm_3/while/lstm_cell_3/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
'lstm_3/while/lstm_cell_3/dropout_4/CastCast3lstm_3/while/lstm_cell_3/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬»
(lstm_3/while/lstm_cell_3/dropout_4/Mul_1Mul*lstm_3/while/lstm_cell_3/dropout_4/Mul:z:0+lstm_3/while/lstm_cell_3/dropout_4/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
(lstm_3/while/lstm_cell_3/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Â
&lstm_3/while/lstm_cell_3/dropout_5/MulMul-lstm_3/while/lstm_cell_3/ones_like_1:output:01lstm_3/while/lstm_cell_3/dropout_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
(lstm_3/while/lstm_cell_3/dropout_5/ShapeShape-lstm_3/while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:Ã
?lstm_3/while/lstm_cell_3/dropout_5/random_uniform/RandomUniformRandomUniform1lstm_3/while/lstm_cell_3/dropout_5/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0v
1lstm_3/while/lstm_cell_3/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>ø
/lstm_3/while/lstm_cell_3/dropout_5/GreaterEqualGreaterEqualHlstm_3/while/lstm_cell_3/dropout_5/random_uniform/RandomUniform:output:0:lstm_3/while/lstm_cell_3/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
'lstm_3/while/lstm_cell_3/dropout_5/CastCast3lstm_3/while/lstm_cell_3/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬»
(lstm_3/while/lstm_cell_3/dropout_5/Mul_1Mul*lstm_3/while/lstm_cell_3/dropout_5/Mul:z:0+lstm_3/while/lstm_cell_3/dropout_5/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
(lstm_3/while/lstm_cell_3/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Â
&lstm_3/while/lstm_cell_3/dropout_6/MulMul-lstm_3/while/lstm_cell_3/ones_like_1:output:01lstm_3/while/lstm_cell_3/dropout_6/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
(lstm_3/while/lstm_cell_3/dropout_6/ShapeShape-lstm_3/while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:Ã
?lstm_3/while/lstm_cell_3/dropout_6/random_uniform/RandomUniformRandomUniform1lstm_3/while/lstm_cell_3/dropout_6/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0v
1lstm_3/while/lstm_cell_3/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>ø
/lstm_3/while/lstm_cell_3/dropout_6/GreaterEqualGreaterEqualHlstm_3/while/lstm_cell_3/dropout_6/random_uniform/RandomUniform:output:0:lstm_3/while/lstm_cell_3/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
'lstm_3/while/lstm_cell_3/dropout_6/CastCast3lstm_3/while/lstm_cell_3/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬»
(lstm_3/while/lstm_cell_3/dropout_6/Mul_1Mul*lstm_3/while/lstm_cell_3/dropout_6/Mul:z:0+lstm_3/while/lstm_cell_3/dropout_6/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
(lstm_3/while/lstm_cell_3/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Â
&lstm_3/while/lstm_cell_3/dropout_7/MulMul-lstm_3/while/lstm_cell_3/ones_like_1:output:01lstm_3/while/lstm_cell_3/dropout_7/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
(lstm_3/while/lstm_cell_3/dropout_7/ShapeShape-lstm_3/while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:Ã
?lstm_3/while/lstm_cell_3/dropout_7/random_uniform/RandomUniformRandomUniform1lstm_3/while/lstm_cell_3/dropout_7/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0v
1lstm_3/while/lstm_cell_3/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>ø
/lstm_3/while/lstm_cell_3/dropout_7/GreaterEqualGreaterEqualHlstm_3/while/lstm_cell_3/dropout_7/random_uniform/RandomUniform:output:0:lstm_3/while/lstm_cell_3/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
'lstm_3/while/lstm_cell_3/dropout_7/CastCast3lstm_3/while/lstm_cell_3/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬»
(lstm_3/while/lstm_cell_3/dropout_7/Mul_1Mul*lstm_3/while/lstm_cell_3/dropout_7/Mul:z:0+lstm_3/while/lstm_cell_3/dropout_7/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬»
lstm_3/while/lstm_cell_3/mulMul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0*lstm_3/while/lstm_cell_3/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¿
lstm_3/while/lstm_cell_3/mul_1Mul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0,lstm_3/while/lstm_cell_3/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¿
lstm_3/while/lstm_cell_3/mul_2Mul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0,lstm_3/while/lstm_cell_3/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¿
lstm_3/while/lstm_cell_3/mul_3Mul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0,lstm_3/while/lstm_cell_3/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈj
(lstm_3/while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¨
-lstm_3/while/lstm_cell_3/split/ReadVariableOpReadVariableOp8lstm_3_while_lstm_cell_3_split_readvariableop_resource_0* 
_output_shapes
:
È°	*
dtype0ñ
lstm_3/while/lstm_cell_3/splitSplit1lstm_3/while/lstm_cell_3/split/split_dim:output:05lstm_3/while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È¬:
È¬:
È¬:
È¬*
	num_split§
lstm_3/while/lstm_cell_3/MatMulMatMul lstm_3/while/lstm_cell_3/mul:z:0'lstm_3/while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬«
!lstm_3/while/lstm_cell_3/MatMul_1MatMul"lstm_3/while/lstm_cell_3/mul_1:z:0'lstm_3/while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬«
!lstm_3/while/lstm_cell_3/MatMul_2MatMul"lstm_3/while/lstm_cell_3/mul_2:z:0'lstm_3/while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬«
!lstm_3/while/lstm_cell_3/MatMul_3MatMul"lstm_3/while/lstm_cell_3/mul_3:z:0'lstm_3/while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
*lstm_3/while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : §
/lstm_3/while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp:lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:°	*
dtype0ã
 lstm_3/while/lstm_cell_3/split_1Split3lstm_3/while/lstm_cell_3/split_1/split_dim:output:07lstm_3/while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:¬:¬:¬:¬*
	num_split´
 lstm_3/while/lstm_cell_3/BiasAddBiasAdd)lstm_3/while/lstm_cell_3/MatMul:product:0)lstm_3/while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¸
"lstm_3/while/lstm_cell_3/BiasAdd_1BiasAdd+lstm_3/while/lstm_cell_3/MatMul_1:product:0)lstm_3/while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¸
"lstm_3/while/lstm_cell_3/BiasAdd_2BiasAdd+lstm_3/while/lstm_cell_3/MatMul_2:product:0)lstm_3/while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¸
"lstm_3/while/lstm_cell_3/BiasAdd_3BiasAdd+lstm_3/while/lstm_cell_3/MatMul_3:product:0)lstm_3/while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¢
lstm_3/while/lstm_cell_3/mul_4Mullstm_3_while_placeholder_2,lstm_3/while/lstm_cell_3/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¢
lstm_3/while/lstm_cell_3/mul_5Mullstm_3_while_placeholder_2,lstm_3/while/lstm_cell_3/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¢
lstm_3/while/lstm_cell_3/mul_6Mullstm_3_while_placeholder_2,lstm_3/while/lstm_cell_3/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¢
lstm_3/while/lstm_cell_3/mul_7Mullstm_3_while_placeholder_2,lstm_3/while/lstm_cell_3/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
'lstm_3/while/lstm_cell_3/ReadVariableOpReadVariableOp2lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0}
,lstm_3/while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.lstm_3/while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  
.lstm_3/while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ê
&lstm_3/while/lstm_cell_3/strided_sliceStridedSlice/lstm_3/while/lstm_cell_3/ReadVariableOp:value:05lstm_3/while/lstm_cell_3/strided_slice/stack:output:07lstm_3/while/lstm_cell_3/strided_slice/stack_1:output:07lstm_3/while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask³
!lstm_3/while/lstm_cell_3/MatMul_4MatMul"lstm_3/while/lstm_cell_3/mul_4:z:0/lstm_3/while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬°
lstm_3/while/lstm_cell_3/addAddV2)lstm_3/while/lstm_cell_3/BiasAdd:output:0+lstm_3/while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 lstm_3/while/lstm_cell_3/SigmoidSigmoid lstm_3/while/lstm_cell_3/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
)lstm_3/while/lstm_cell_3/ReadVariableOp_1ReadVariableOp2lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0
.lstm_3/while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  
0lstm_3/while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  
0lstm_3/while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
(lstm_3/while/lstm_cell_3/strided_slice_1StridedSlice1lstm_3/while/lstm_cell_3/ReadVariableOp_1:value:07lstm_3/while/lstm_cell_3/strided_slice_1/stack:output:09lstm_3/while/lstm_cell_3/strided_slice_1/stack_1:output:09lstm_3/while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskµ
!lstm_3/while/lstm_cell_3/MatMul_5MatMul"lstm_3/while/lstm_cell_3/mul_5:z:01lstm_3/while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬´
lstm_3/while/lstm_cell_3/add_1AddV2+lstm_3/while/lstm_cell_3/BiasAdd_1:output:0+lstm_3/while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"lstm_3/while/lstm_cell_3/Sigmoid_1Sigmoid"lstm_3/while/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/while/lstm_cell_3/mul_8Mul&lstm_3/while/lstm_cell_3/Sigmoid_1:y:0lstm_3_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
)lstm_3/while/lstm_cell_3/ReadVariableOp_2ReadVariableOp2lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0
.lstm_3/while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  
0lstm_3/while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
0lstm_3/while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
(lstm_3/while/lstm_cell_3/strided_slice_2StridedSlice1lstm_3/while/lstm_cell_3/ReadVariableOp_2:value:07lstm_3/while/lstm_cell_3/strided_slice_2/stack:output:09lstm_3/while/lstm_cell_3/strided_slice_2/stack_1:output:09lstm_3/while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskµ
!lstm_3/while/lstm_cell_3/MatMul_6MatMul"lstm_3/while/lstm_cell_3/mul_6:z:01lstm_3/while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬´
lstm_3/while/lstm_cell_3/add_2AddV2+lstm_3/while/lstm_cell_3/BiasAdd_2:output:0+lstm_3/while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬|
lstm_3/while/lstm_cell_3/TanhTanh"lstm_3/while/lstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¡
lstm_3/while/lstm_cell_3/mul_9Mul$lstm_3/while/lstm_cell_3/Sigmoid:y:0!lstm_3/while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¢
lstm_3/while/lstm_cell_3/add_3AddV2"lstm_3/while/lstm_cell_3/mul_8:z:0"lstm_3/while/lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
)lstm_3/while/lstm_cell_3/ReadVariableOp_3ReadVariableOp2lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0
.lstm_3/while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
0lstm_3/while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
0lstm_3/while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
(lstm_3/while/lstm_cell_3/strided_slice_3StridedSlice1lstm_3/while/lstm_cell_3/ReadVariableOp_3:value:07lstm_3/while/lstm_cell_3/strided_slice_3/stack:output:09lstm_3/while/lstm_cell_3/strided_slice_3/stack_1:output:09lstm_3/while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskµ
!lstm_3/while/lstm_cell_3/MatMul_7MatMul"lstm_3/while/lstm_cell_3/mul_7:z:01lstm_3/while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬´
lstm_3/while/lstm_cell_3/add_4AddV2+lstm_3/while/lstm_cell_3/BiasAdd_3:output:0+lstm_3/while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"lstm_3/while/lstm_cell_3/Sigmoid_2Sigmoid"lstm_3/while/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬~
lstm_3/while/lstm_cell_3/Tanh_1Tanh"lstm_3/while/lstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
lstm_3/while/lstm_cell_3/mul_10Mul&lstm_3/while/lstm_cell_3/Sigmoid_2:y:0#lstm_3/while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬á
1lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_3_while_placeholder_1lstm_3_while_placeholder#lstm_3/while/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒT
lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_3/while/addAddV2lstm_3_while_placeholderlstm_3/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_3/while/add_1AddV2&lstm_3_while_lstm_3_while_loop_counterlstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_3/while/IdentityIdentitylstm_3/while/add_1:z:0^lstm_3/while/NoOp*
T0*
_output_shapes
: 
lstm_3/while/Identity_1Identity,lstm_3_while_lstm_3_while_maximum_iterations^lstm_3/while/NoOp*
T0*
_output_shapes
: n
lstm_3/while/Identity_2Identitylstm_3/while/add:z:0^lstm_3/while/NoOp*
T0*
_output_shapes
: ®
lstm_3/while/Identity_3IdentityAlstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_3/while/NoOp*
T0*
_output_shapes
: :éèÒ
lstm_3/while/Identity_4Identity#lstm_3/while/lstm_cell_3/mul_10:z:0^lstm_3/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/while/Identity_5Identity"lstm_3/while/lstm_cell_3/add_3:z:0^lstm_3/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ã
lstm_3/while/NoOpNoOp(^lstm_3/while/lstm_cell_3/ReadVariableOp*^lstm_3/while/lstm_cell_3/ReadVariableOp_1*^lstm_3/while/lstm_cell_3/ReadVariableOp_2*^lstm_3/while/lstm_cell_3/ReadVariableOp_3.^lstm_3/while/lstm_cell_3/split/ReadVariableOp0^lstm_3/while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_3_while_identitylstm_3/while/Identity:output:0";
lstm_3_while_identity_1 lstm_3/while/Identity_1:output:0";
lstm_3_while_identity_2 lstm_3/while/Identity_2:output:0";
lstm_3_while_identity_3 lstm_3/while/Identity_3:output:0";
lstm_3_while_identity_4 lstm_3/while/Identity_4:output:0";
lstm_3_while_identity_5 lstm_3/while/Identity_5:output:0"H
!lstm_3_while_lstm_3_strided_slice#lstm_3_while_lstm_3_strided_slice_0"f
0lstm_3_while_lstm_cell_3_readvariableop_resource2lstm_3_while_lstm_cell_3_readvariableop_resource_0"v
8lstm_3_while_lstm_cell_3_split_1_readvariableop_resource:lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0"r
6lstm_3_while_lstm_cell_3_split_readvariableop_resource8lstm_3_while_lstm_cell_3_split_readvariableop_resource_0"Ä
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensoralstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : 2R
'lstm_3/while/lstm_cell_3/ReadVariableOp'lstm_3/while/lstm_cell_3/ReadVariableOp2V
)lstm_3/while/lstm_cell_3/ReadVariableOp_1)lstm_3/while/lstm_cell_3/ReadVariableOp_12V
)lstm_3/while/lstm_cell_3/ReadVariableOp_2)lstm_3/while/lstm_cell_3/ReadVariableOp_22V
)lstm_3/while/lstm_cell_3/ReadVariableOp_3)lstm_3/while/lstm_cell_3/ReadVariableOp_32^
-lstm_3/while/lstm_cell_3/split/ReadVariableOp-lstm_3/while/lstm_cell_3/split/ReadVariableOp2b
/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
: 
°
¼
while_cond_99023
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_99023___redundant_placeholder03
/while_while_cond_99023___redundant_placeholder13
/while_while_cond_99023___redundant_placeholder23
/while_while_cond_99023___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
:
ú©
¹
B__inference_model_3_layer_call_and_return_conditional_losses_99710
inputs_0
inputs_1
inputs_2
inputs_36
"embedding_1_embedding_lookup_99456:
¸dÈD
0lstm_3_lstm_cell_3_split_readvariableop_resource:
È°	A
2lstm_3_lstm_cell_3_split_1_readvariableop_resource:	°	>
*lstm_3_lstm_cell_3_readvariableop_resource:
¬°	I
5time_distributed_dense_matmul_readvariableop_resource:
¬¸dE
6time_distributed_dense_biasadd_readvariableop_resource:	¸d
identity

identity_1

identity_2¢embedding_1/embedding_lookup¢!lstm_3/lstm_cell_3/ReadVariableOp¢#lstm_3/lstm_cell_3/ReadVariableOp_1¢#lstm_3/lstm_cell_3/ReadVariableOp_2¢#lstm_3/lstm_cell_3/ReadVariableOp_3¢'lstm_3/lstm_cell_3/split/ReadVariableOp¢)lstm_3/lstm_cell_3/split_1/ReadVariableOp¢lstm_3/while¢-time_distributed/dense/BiasAdd/ReadVariableOp¢,time_distributed/dense/MatMul/ReadVariableOpl
embedding_1/CastCastinputs_0*

DstT0*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿó
embedding_1/embedding_lookupResourceGather"embedding_1_embedding_lookup_99456embedding_1/Cast:y:0*
Tindices0*5
_class+
)'loc:@embedding_1/embedding_lookup/99456*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*
dtype0Ï
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_1/embedding_lookup/99456*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ£
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈj
lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¯
lstm_3/transpose	Transpose0embedding_1/embedding_lookup/Identity_1:output:0lstm_3/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈP
lstm_3/ShapeShapelstm_3/transpose:y:0*
T0*
_output_shapes
:d
lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm_3/strided_sliceStridedSlicelstm_3/Shape:output:0#lstm_3/strided_slice/stack:output:0%lstm_3/strided_slice/stack_1:output:0%lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
lstm_3/TensorArrayV2TensorListReserve+lstm_3/TensorArrayV2/element_shape:output:0lstm_3/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   õ
.lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_3/transpose:y:0Elstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒf
lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_3/strided_slice_1StridedSlicelstm_3/transpose:y:0%lstm_3/strided_slice_1/stack:output:0'lstm_3/strided_slice_1/stack_1:output:0'lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_maskq
"lstm_3/lstm_cell_3/ones_like/ShapeShapelstm_3/strided_slice_1:output:0*
T0*
_output_shapes
:g
"lstm_3/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?±
lstm_3/lstm_cell_3/ones_likeFill+lstm_3/lstm_cell_3/ones_like/Shape:output:0+lstm_3/lstm_cell_3/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ\
$lstm_3/lstm_cell_3/ones_like_1/ShapeShapeinputs_2*
T0*
_output_shapes
:i
$lstm_3/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?·
lstm_3/lstm_cell_3/ones_like_1Fill-lstm_3/lstm_cell_3/ones_like_1/Shape:output:0-lstm_3/lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/lstm_cell_3/mulMullstm_3/strided_slice_1:output:0%lstm_3/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_3/lstm_cell_3/mul_1Mullstm_3/strided_slice_1:output:0%lstm_3/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_3/lstm_cell_3/mul_2Mullstm_3/strided_slice_1:output:0%lstm_3/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_3/lstm_cell_3/mul_3Mullstm_3/strided_slice_1:output:0%lstm_3/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈd
"lstm_3/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
'lstm_3/lstm_cell_3/split/ReadVariableOpReadVariableOp0lstm_3_lstm_cell_3_split_readvariableop_resource* 
_output_shapes
:
È°	*
dtype0ß
lstm_3/lstm_cell_3/splitSplit+lstm_3/lstm_cell_3/split/split_dim:output:0/lstm_3/lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È¬:
È¬:
È¬:
È¬*
	num_split
lstm_3/lstm_cell_3/MatMulMatMullstm_3/lstm_cell_3/mul:z:0!lstm_3/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/lstm_cell_3/MatMul_1MatMullstm_3/lstm_cell_3/mul_1:z:0!lstm_3/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/lstm_cell_3/MatMul_2MatMullstm_3/lstm_cell_3/mul_2:z:0!lstm_3/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/lstm_cell_3/MatMul_3MatMullstm_3/lstm_cell_3/mul_3:z:0!lstm_3/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
$lstm_3/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
)lstm_3/lstm_cell_3/split_1/ReadVariableOpReadVariableOp2lstm_3_lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:°	*
dtype0Ñ
lstm_3/lstm_cell_3/split_1Split-lstm_3/lstm_cell_3/split_1/split_dim:output:01lstm_3/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:¬:¬:¬:¬*
	num_split¢
lstm_3/lstm_cell_3/BiasAddBiasAdd#lstm_3/lstm_cell_3/MatMul:product:0#lstm_3/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
lstm_3/lstm_cell_3/BiasAdd_1BiasAdd%lstm_3/lstm_cell_3/MatMul_1:product:0#lstm_3/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
lstm_3/lstm_cell_3/BiasAdd_2BiasAdd%lstm_3/lstm_cell_3/MatMul_2:product:0#lstm_3/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
lstm_3/lstm_cell_3/BiasAdd_3BiasAdd%lstm_3/lstm_cell_3/MatMul_3:product:0#lstm_3/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/lstm_cell_3/mul_4Mulinputs_2'lstm_3/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/lstm_cell_3/mul_5Mulinputs_2'lstm_3/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/lstm_cell_3/mul_6Mulinputs_2'lstm_3/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/lstm_cell_3/mul_7Mulinputs_2'lstm_3/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
!lstm_3/lstm_cell_3/ReadVariableOpReadVariableOp*lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0w
&lstm_3/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(lstm_3/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  y
(lstm_3/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ì
 lstm_3/lstm_cell_3/strided_sliceStridedSlice)lstm_3/lstm_cell_3/ReadVariableOp:value:0/lstm_3/lstm_cell_3/strided_slice/stack:output:01lstm_3/lstm_cell_3/strided_slice/stack_1:output:01lstm_3/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask¡
lstm_3/lstm_cell_3/MatMul_4MatMullstm_3/lstm_cell_3/mul_4:z:0)lstm_3/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/lstm_cell_3/addAddV2#lstm_3/lstm_cell_3/BiasAdd:output:0%lstm_3/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬t
lstm_3/lstm_cell_3/SigmoidSigmoidlstm_3/lstm_cell_3/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
#lstm_3/lstm_cell_3/ReadVariableOp_1ReadVariableOp*lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0y
(lstm_3/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  {
*lstm_3/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  {
*lstm_3/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm_3/lstm_cell_3/strided_slice_1StridedSlice+lstm_3/lstm_cell_3/ReadVariableOp_1:value:01lstm_3/lstm_cell_3/strided_slice_1/stack:output:03lstm_3/lstm_cell_3/strided_slice_1/stack_1:output:03lstm_3/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask£
lstm_3/lstm_cell_3/MatMul_5MatMullstm_3/lstm_cell_3/mul_5:z:0+lstm_3/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¢
lstm_3/lstm_cell_3/add_1AddV2%lstm_3/lstm_cell_3/BiasAdd_1:output:0%lstm_3/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬x
lstm_3/lstm_cell_3/Sigmoid_1Sigmoidlstm_3/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬~
lstm_3/lstm_cell_3/mul_8Mul lstm_3/lstm_cell_3/Sigmoid_1:y:0inputs_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
#lstm_3/lstm_cell_3/ReadVariableOp_2ReadVariableOp*lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0y
(lstm_3/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  {
*lstm_3/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      {
*lstm_3/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm_3/lstm_cell_3/strided_slice_2StridedSlice+lstm_3/lstm_cell_3/ReadVariableOp_2:value:01lstm_3/lstm_cell_3/strided_slice_2/stack:output:03lstm_3/lstm_cell_3/strided_slice_2/stack_1:output:03lstm_3/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask£
lstm_3/lstm_cell_3/MatMul_6MatMullstm_3/lstm_cell_3/mul_6:z:0+lstm_3/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¢
lstm_3/lstm_cell_3/add_2AddV2%lstm_3/lstm_cell_3/BiasAdd_2:output:0%lstm_3/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬p
lstm_3/lstm_cell_3/TanhTanhlstm_3/lstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/lstm_cell_3/mul_9Mullstm_3/lstm_cell_3/Sigmoid:y:0lstm_3/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/lstm_cell_3/add_3AddV2lstm_3/lstm_cell_3/mul_8:z:0lstm_3/lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
#lstm_3/lstm_cell_3/ReadVariableOp_3ReadVariableOp*lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0y
(lstm_3/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      {
*lstm_3/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_3/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
"lstm_3/lstm_cell_3/strided_slice_3StridedSlice+lstm_3/lstm_cell_3/ReadVariableOp_3:value:01lstm_3/lstm_cell_3/strided_slice_3/stack:output:03lstm_3/lstm_cell_3/strided_slice_3/stack_1:output:03lstm_3/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask£
lstm_3/lstm_cell_3/MatMul_7MatMullstm_3/lstm_cell_3/mul_7:z:0+lstm_3/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¢
lstm_3/lstm_cell_3/add_4AddV2%lstm_3/lstm_cell_3/BiasAdd_3:output:0%lstm_3/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬x
lstm_3/lstm_cell_3/Sigmoid_2Sigmoidlstm_3/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r
lstm_3/lstm_cell_3/Tanh_1Tanhlstm_3/lstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/lstm_cell_3/mul_10Mul lstm_3/lstm_cell_3/Sigmoid_2:y:0lstm_3/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬u
$lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  Ë
lstm_3/TensorArrayV2_1TensorListReserve-lstm_3/TensorArrayV2_1/element_shape:output:0lstm_3/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒM
lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : »
lstm_3/whileWhile"lstm_3/while/loop_counter:output:0(lstm_3/while/maximum_iterations:output:0lstm_3/time:output:0lstm_3/TensorArrayV2_1:handle:0inputs_2inputs_3lstm_3/strided_slice:output:0>lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:00lstm_3_lstm_cell_3_split_readvariableop_resource2lstm_3_lstm_cell_3_split_1_readvariableop_resource*lstm_3_lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_3_while_body_99554*#
condR
lstm_3_while_cond_99553*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *
parallel_iterations 
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  á
)lstm_3/TensorArrayV2Stack/TensorListStackTensorListStacklstm_3/while:output:3@lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬*
element_dtype0o
lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿh
lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
lstm_3/strided_slice_2StridedSlice2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_3/strided_slice_2/stack:output:0'lstm_3/strided_slice_2/stack_1:output:0'lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_maskl
lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          µ
lstm_3/transpose_1	Transpose2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_3/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬b
lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    \
time_distributed/ShapeShapelstm_3/transpose_1:y:0*
T0*
_output_shapes
:n
$time_distributed/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:p
&time_distributed/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&time_distributed/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
time_distributed/strided_sliceStridedSlicetime_distributed/Shape:output:0-time_distributed/strided_slice/stack:output:0/time_distributed/strided_slice/stack_1:output:0/time_distributed/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  
time_distributed/ReshapeReshapelstm_3/transpose_1:y:0'time_distributed/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¤
,time_distributed/dense/MatMul/ReadVariableOpReadVariableOp5time_distributed_dense_matmul_readvariableop_resource* 
_output_shapes
:
¬¸d*
dtype0³
time_distributed/dense/MatMulMatMul!time_distributed/Reshape:output:04time_distributed/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸d¡
-time_distributed/dense/BiasAdd/ReadVariableOpReadVariableOp6time_distributed_dense_biasadd_readvariableop_resource*
_output_shapes	
:¸d*
dtype0¼
time_distributed/dense/BiasAddBiasAdd'time_distributed/dense/MatMul:product:05time_distributed/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸d
time_distributed/dense/SoftmaxSoftmax'time_distributed/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸dm
"time_distributed/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿe
"time_distributed/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :¸dÙ
 time_distributed/Reshape_1/shapePack+time_distributed/Reshape_1/shape/0:output:0'time_distributed/strided_slice:output:0+time_distributed/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:º
time_distributed/Reshape_1Reshape(time_distributed/dense/Softmax:softmax:0)time_distributed/Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸dq
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  
time_distributed/Reshape_2Reshapelstm_3/transpose_1:y:0)time_distributed/Reshape_2/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
IdentityIdentity#time_distributed/Reshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸dg

Identity_1Identitylstm_3/while:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬g

Identity_2Identitylstm_3/while:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¿
NoOpNoOp^embedding_1/embedding_lookup"^lstm_3/lstm_cell_3/ReadVariableOp$^lstm_3/lstm_cell_3/ReadVariableOp_1$^lstm_3/lstm_cell_3/ReadVariableOp_2$^lstm_3/lstm_cell_3/ReadVariableOp_3(^lstm_3/lstm_cell_3/split/ReadVariableOp*^lstm_3/lstm_cell_3/split_1/ReadVariableOp^lstm_3/while.^time_distributed/dense/BiasAdd/ReadVariableOp-^time_distributed/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ>¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : : 2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2F
!lstm_3/lstm_cell_3/ReadVariableOp!lstm_3/lstm_cell_3/ReadVariableOp2J
#lstm_3/lstm_cell_3/ReadVariableOp_1#lstm_3/lstm_cell_3/ReadVariableOp_12J
#lstm_3/lstm_cell_3/ReadVariableOp_2#lstm_3/lstm_cell_3/ReadVariableOp_22J
#lstm_3/lstm_cell_3/ReadVariableOp_3#lstm_3/lstm_cell_3/ReadVariableOp_32R
'lstm_3/lstm_cell_3/split/ReadVariableOp'lstm_3/lstm_cell_3/split/ReadVariableOp2V
)lstm_3/lstm_cell_3/split_1/ReadVariableOp)lstm_3/lstm_cell_3/split_1/ReadVariableOp2
lstm_3/whilelstm_3/while2^
-time_distributed/dense/BiasAdd/ReadVariableOp-time_distributed/dense/BiasAdd/ReadVariableOp2\
,time_distributed/dense/MatMul/ReadVariableOp,time_distributed/dense/MatMul/ReadVariableOp:Z V
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>¬
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
inputs/2:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
inputs/3
¢
Ö
'__inference_model_3_layer_call_fn_99345
input_2
input_7
input_5
input_6
unknown:
¸dÈ
	unknown_0:
È°	
	unknown_1:	°	
	unknown_2:
¬°	
	unknown_3:
¬¸d
	unknown_4:	¸d
identity

identity_1

identity_2¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinput_2input_7input_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2
*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*(
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_99302}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸dr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ>¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:UQ
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>¬
!
_user_specified_name	input_7:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
!
_user_specified_name	input_5:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
!
_user_specified_name	input_6
ô	
È
lstm_3_while_cond_99553*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3*
&lstm_3_while_less_lstm_3_strided_sliceA
=lstm_3_while_lstm_3_while_cond_99553___redundant_placeholder0A
=lstm_3_while_lstm_3_while_cond_99553___redundant_placeholder1A
=lstm_3_while_lstm_3_while_cond_99553___redundant_placeholder2A
=lstm_3_while_lstm_3_while_cond_99553___redundant_placeholder3
lstm_3_while_identity
|
lstm_3/while/LessLesslstm_3_while_placeholder&lstm_3_while_less_lstm_3_strided_slice*
T0*
_output_shapes
: Y
lstm_3/while/IdentityIdentitylstm_3/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_3_while_identitylstm_3/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
:
ý
Ó
$__inference_signature_wrapper_100125
input_2
input_5
input_6
input_7
unknown:
¸dÈ
	unknown_0:
È°	
	unknown_1:	°	
	unknown_2:
¬°	
	unknown_3:
¬¸d
	unknown_4:	¸d
identity

identity_1

identity_2¢StatefulPartitionedCallÃ
StatefulPartitionedCallStatefulPartitionedCallinput_2input_7input_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2
*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d*(
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_97934p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ>¬: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
!
_user_specified_name	input_5:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
!
_user_specified_name	input_6:UQ
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>¬
!
_user_specified_name	input_7
ù
÷
,__inference_lstm_cell_3_layer_call_fn_101516

inputs
states_0
states_1
unknown:
È°	
	unknown_0:	°	
	unknown_1:
¬°	
identity

identity_1

identity_2¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_98317p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
states/1
µ
Á
while_cond_100921
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_100921___redundant_placeholder04
0while_while_cond_100921___redundant_placeholder14
0while_while_cond_100921___redundant_placeholder24
0while_while_cond_100921___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
:
·

L__inference_time_distributed_layer_call_and_return_conditional_losses_101482

inputs8
$dense_matmul_readvariableop_resource:
¬¸d4
%dense_biasadd_readvariableop_resource:	¸d
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
¬¸d*
dtype0
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸d
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:¸d*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸dc
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸d\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :¸d
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapedense/Softmax:softmax:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸do
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
Ç
ö
B__inference_model_3_layer_call_and_return_conditional_losses_99373
input_2
input_7
input_5
input_6%
embedding_1_99351:
¸dÈ 
lstm_3_99354:
È°	
lstm_3_99356:	°	 
lstm_3_99358:
¬°	*
time_distributed_99363:
¬¸d%
time_distributed_99365:	¸d
identity

identity_1

identity_2¢#embedding_1/StatefulPartitionedCall¢lstm_3/StatefulPartitionedCall¢(time_distributed/StatefulPartitionedCallö
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinput_2embedding_1_99351*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_98565ê
lstm_3/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0input_5input_6lstm_3_99354lstm_3_99356lstm_3_99358*
Tin

2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_98802¿
(time_distributed/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0time_distributed_99363time_distributed_99365*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_98496o
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  ¨
time_distributed/ReshapeReshape'lstm_3/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
IdentityIdentity1time_distributed/StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸dy

Identity_1Identity'lstm_3/StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬y

Identity_2Identity'lstm_3/StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¸
NoOpNoOp$^embedding_1/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ>¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : : 2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall:Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:UQ
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>¬
!
_user_specified_name	input_7:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
!
_user_specified_name	input_5:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
!
_user_specified_name	input_6
·:

A__inference_lstm_3_layer_call_and_return_conditional_losses_98447

inputs%
lstm_cell_3_98363:
È°	 
lstm_cell_3_98365:	°	%
lstm_cell_3_98367:
¬°	
identity

identity_1

identity_2¢#lstm_cell_3/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¬s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¬w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_maskò
#lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_3_98363lstm_cell_3_98365lstm_cell_3_98367*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_98317n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ³
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_3_98363lstm_cell_3_98365lstm_cell_3_98367*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_98376*
condR
while_cond_98375*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬t
NoOpNoOp$^lstm_cell_3/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ: : : 2J
#lstm_cell_3/StatefulPartitionedCall#lstm_cell_3/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
®
Ú
'__inference_model_3_layer_call_fn_99449
inputs_0
inputs_1
inputs_2
inputs_3
unknown:
¸dÈ
	unknown_0:
È°	
	unknown_1:	°	
	unknown_2:
¬°	
	unknown_3:
¬¸d
	unknown_4:	¸d
identity

identity_1

identity_2¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2
*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*(
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_99302}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸dr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ>¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>¬
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
inputs/2:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
inputs/3
Å

&__inference_dense_layer_call_fn_101753

inputs
unknown:
¬¸d
	unknown_0:	¸d
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_98485p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs


lstm_3_while_body_99554*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3'
#lstm_3_while_lstm_3_strided_slice_0e
alstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0L
8lstm_3_while_lstm_cell_3_split_readvariableop_resource_0:
È°	I
:lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0:	°	F
2lstm_3_while_lstm_cell_3_readvariableop_resource_0:
¬°	
lstm_3_while_identity
lstm_3_while_identity_1
lstm_3_while_identity_2
lstm_3_while_identity_3
lstm_3_while_identity_4
lstm_3_while_identity_5%
!lstm_3_while_lstm_3_strided_slicec
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensorJ
6lstm_3_while_lstm_cell_3_split_readvariableop_resource:
È°	G
8lstm_3_while_lstm_cell_3_split_1_readvariableop_resource:	°	D
0lstm_3_while_lstm_cell_3_readvariableop_resource:
¬°	¢'lstm_3/while/lstm_cell_3/ReadVariableOp¢)lstm_3/while/lstm_cell_3/ReadVariableOp_1¢)lstm_3/while/lstm_cell_3/ReadVariableOp_2¢)lstm_3/while/lstm_cell_3/ReadVariableOp_3¢-lstm_3/while/lstm_cell_3/split/ReadVariableOp¢/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   Ê
0lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0lstm_3_while_placeholderGlstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype0
(lstm_3/while/lstm_cell_3/ones_like/ShapeShape7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:m
(lstm_3/while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ã
"lstm_3/while/lstm_cell_3/ones_likeFill1lstm_3/while/lstm_cell_3/ones_like/Shape:output:01lstm_3/while/lstm_cell_3/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈt
*lstm_3/while/lstm_cell_3/ones_like_1/ShapeShapelstm_3_while_placeholder_2*
T0*
_output_shapes
:o
*lstm_3/while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?É
$lstm_3/while/lstm_cell_3/ones_like_1Fill3lstm_3/while/lstm_cell_3/ones_like_1/Shape:output:03lstm_3/while/lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¼
lstm_3/while/lstm_cell_3/mulMul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_3/while/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¾
lstm_3/while/lstm_cell_3/mul_1Mul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_3/while/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¾
lstm_3/while/lstm_cell_3/mul_2Mul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_3/while/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¾
lstm_3/while/lstm_cell_3/mul_3Mul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_3/while/lstm_cell_3/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈj
(lstm_3/while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¨
-lstm_3/while/lstm_cell_3/split/ReadVariableOpReadVariableOp8lstm_3_while_lstm_cell_3_split_readvariableop_resource_0* 
_output_shapes
:
È°	*
dtype0ñ
lstm_3/while/lstm_cell_3/splitSplit1lstm_3/while/lstm_cell_3/split/split_dim:output:05lstm_3/while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È¬:
È¬:
È¬:
È¬*
	num_split§
lstm_3/while/lstm_cell_3/MatMulMatMul lstm_3/while/lstm_cell_3/mul:z:0'lstm_3/while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬«
!lstm_3/while/lstm_cell_3/MatMul_1MatMul"lstm_3/while/lstm_cell_3/mul_1:z:0'lstm_3/while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬«
!lstm_3/while/lstm_cell_3/MatMul_2MatMul"lstm_3/while/lstm_cell_3/mul_2:z:0'lstm_3/while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬«
!lstm_3/while/lstm_cell_3/MatMul_3MatMul"lstm_3/while/lstm_cell_3/mul_3:z:0'lstm_3/while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
*lstm_3/while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : §
/lstm_3/while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp:lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:°	*
dtype0ã
 lstm_3/while/lstm_cell_3/split_1Split3lstm_3/while/lstm_cell_3/split_1/split_dim:output:07lstm_3/while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:¬:¬:¬:¬*
	num_split´
 lstm_3/while/lstm_cell_3/BiasAddBiasAdd)lstm_3/while/lstm_cell_3/MatMul:product:0)lstm_3/while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¸
"lstm_3/while/lstm_cell_3/BiasAdd_1BiasAdd+lstm_3/while/lstm_cell_3/MatMul_1:product:0)lstm_3/while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¸
"lstm_3/while/lstm_cell_3/BiasAdd_2BiasAdd+lstm_3/while/lstm_cell_3/MatMul_2:product:0)lstm_3/while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¸
"lstm_3/while/lstm_cell_3/BiasAdd_3BiasAdd+lstm_3/while/lstm_cell_3/MatMul_3:product:0)lstm_3/while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
lstm_3/while/lstm_cell_3/mul_4Mullstm_3_while_placeholder_2-lstm_3/while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
lstm_3/while/lstm_cell_3/mul_5Mullstm_3_while_placeholder_2-lstm_3/while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
lstm_3/while/lstm_cell_3/mul_6Mullstm_3_while_placeholder_2-lstm_3/while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
lstm_3/while/lstm_cell_3/mul_7Mullstm_3_while_placeholder_2-lstm_3/while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
'lstm_3/while/lstm_cell_3/ReadVariableOpReadVariableOp2lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0}
,lstm_3/while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.lstm_3/while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  
.lstm_3/while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ê
&lstm_3/while/lstm_cell_3/strided_sliceStridedSlice/lstm_3/while/lstm_cell_3/ReadVariableOp:value:05lstm_3/while/lstm_cell_3/strided_slice/stack:output:07lstm_3/while/lstm_cell_3/strided_slice/stack_1:output:07lstm_3/while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask³
!lstm_3/while/lstm_cell_3/MatMul_4MatMul"lstm_3/while/lstm_cell_3/mul_4:z:0/lstm_3/while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬°
lstm_3/while/lstm_cell_3/addAddV2)lstm_3/while/lstm_cell_3/BiasAdd:output:0+lstm_3/while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 lstm_3/while/lstm_cell_3/SigmoidSigmoid lstm_3/while/lstm_cell_3/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
)lstm_3/while/lstm_cell_3/ReadVariableOp_1ReadVariableOp2lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0
.lstm_3/while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  
0lstm_3/while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  
0lstm_3/while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
(lstm_3/while/lstm_cell_3/strided_slice_1StridedSlice1lstm_3/while/lstm_cell_3/ReadVariableOp_1:value:07lstm_3/while/lstm_cell_3/strided_slice_1/stack:output:09lstm_3/while/lstm_cell_3/strided_slice_1/stack_1:output:09lstm_3/while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskµ
!lstm_3/while/lstm_cell_3/MatMul_5MatMul"lstm_3/while/lstm_cell_3/mul_5:z:01lstm_3/while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬´
lstm_3/while/lstm_cell_3/add_1AddV2+lstm_3/while/lstm_cell_3/BiasAdd_1:output:0+lstm_3/while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"lstm_3/while/lstm_cell_3/Sigmoid_1Sigmoid"lstm_3/while/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/while/lstm_cell_3/mul_8Mul&lstm_3/while/lstm_cell_3/Sigmoid_1:y:0lstm_3_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
)lstm_3/while/lstm_cell_3/ReadVariableOp_2ReadVariableOp2lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0
.lstm_3/while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  
0lstm_3/while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
0lstm_3/while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
(lstm_3/while/lstm_cell_3/strided_slice_2StridedSlice1lstm_3/while/lstm_cell_3/ReadVariableOp_2:value:07lstm_3/while/lstm_cell_3/strided_slice_2/stack:output:09lstm_3/while/lstm_cell_3/strided_slice_2/stack_1:output:09lstm_3/while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskµ
!lstm_3/while/lstm_cell_3/MatMul_6MatMul"lstm_3/while/lstm_cell_3/mul_6:z:01lstm_3/while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬´
lstm_3/while/lstm_cell_3/add_2AddV2+lstm_3/while/lstm_cell_3/BiasAdd_2:output:0+lstm_3/while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬|
lstm_3/while/lstm_cell_3/TanhTanh"lstm_3/while/lstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¡
lstm_3/while/lstm_cell_3/mul_9Mul$lstm_3/while/lstm_cell_3/Sigmoid:y:0!lstm_3/while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¢
lstm_3/while/lstm_cell_3/add_3AddV2"lstm_3/while/lstm_cell_3/mul_8:z:0"lstm_3/while/lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
)lstm_3/while/lstm_cell_3/ReadVariableOp_3ReadVariableOp2lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
¬°	*
dtype0
.lstm_3/while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      
0lstm_3/while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
0lstm_3/while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
(lstm_3/while/lstm_cell_3/strided_slice_3StridedSlice1lstm_3/while/lstm_cell_3/ReadVariableOp_3:value:07lstm_3/while/lstm_cell_3/strided_slice_3/stack:output:09lstm_3/while/lstm_cell_3/strided_slice_3/stack_1:output:09lstm_3/while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_maskµ
!lstm_3/while/lstm_cell_3/MatMul_7MatMul"lstm_3/while/lstm_cell_3/mul_7:z:01lstm_3/while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬´
lstm_3/while/lstm_cell_3/add_4AddV2+lstm_3/while/lstm_cell_3/BiasAdd_3:output:0+lstm_3/while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"lstm_3/while/lstm_cell_3/Sigmoid_2Sigmoid"lstm_3/while/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬~
lstm_3/while/lstm_cell_3/Tanh_1Tanh"lstm_3/while/lstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¦
lstm_3/while/lstm_cell_3/mul_10Mul&lstm_3/while/lstm_cell_3/Sigmoid_2:y:0#lstm_3/while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬á
1lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_3_while_placeholder_1lstm_3_while_placeholder#lstm_3/while/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒT
lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_3/while/addAddV2lstm_3_while_placeholderlstm_3/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_3/while/add_1AddV2&lstm_3_while_lstm_3_while_loop_counterlstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_3/while/IdentityIdentitylstm_3/while/add_1:z:0^lstm_3/while/NoOp*
T0*
_output_shapes
: 
lstm_3/while/Identity_1Identity,lstm_3_while_lstm_3_while_maximum_iterations^lstm_3/while/NoOp*
T0*
_output_shapes
: n
lstm_3/while/Identity_2Identitylstm_3/while/add:z:0^lstm_3/while/NoOp*
T0*
_output_shapes
: ®
lstm_3/while/Identity_3IdentityAlstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_3/while/NoOp*
T0*
_output_shapes
: :éèÒ
lstm_3/while/Identity_4Identity#lstm_3/while/lstm_cell_3/mul_10:z:0^lstm_3/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_3/while/Identity_5Identity"lstm_3/while/lstm_cell_3/add_3:z:0^lstm_3/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ã
lstm_3/while/NoOpNoOp(^lstm_3/while/lstm_cell_3/ReadVariableOp*^lstm_3/while/lstm_cell_3/ReadVariableOp_1*^lstm_3/while/lstm_cell_3/ReadVariableOp_2*^lstm_3/while/lstm_cell_3/ReadVariableOp_3.^lstm_3/while/lstm_cell_3/split/ReadVariableOp0^lstm_3/while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_3_while_identitylstm_3/while/Identity:output:0";
lstm_3_while_identity_1 lstm_3/while/Identity_1:output:0";
lstm_3_while_identity_2 lstm_3/while/Identity_2:output:0";
lstm_3_while_identity_3 lstm_3/while/Identity_3:output:0";
lstm_3_while_identity_4 lstm_3/while/Identity_4:output:0";
lstm_3_while_identity_5 lstm_3/while/Identity_5:output:0"H
!lstm_3_while_lstm_3_strided_slice#lstm_3_while_lstm_3_strided_slice_0"f
0lstm_3_while_lstm_cell_3_readvariableop_resource2lstm_3_while_lstm_cell_3_readvariableop_resource_0"v
8lstm_3_while_lstm_cell_3_split_1_readvariableop_resource:lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0"r
6lstm_3_while_lstm_cell_3_split_readvariableop_resource8lstm_3_while_lstm_cell_3_split_readvariableop_resource_0"Ä
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensoralstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : 2R
'lstm_3/while/lstm_cell_3/ReadVariableOp'lstm_3/while/lstm_cell_3/ReadVariableOp2V
)lstm_3/while/lstm_cell_3/ReadVariableOp_1)lstm_3/while/lstm_cell_3/ReadVariableOp_12V
)lstm_3/while/lstm_cell_3/ReadVariableOp_2)lstm_3/while/lstm_cell_3/ReadVariableOp_22V
)lstm_3/while/lstm_cell_3/ReadVariableOp_3)lstm_3/while/lstm_cell_3/ReadVariableOp_32^
-lstm_3/while/lstm_cell_3/split/ReadVariableOp-lstm_3/while/lstm_cell_3/split/ReadVariableOp2b
/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
: 
©Â
©
B__inference_lstm_3_layer_call_and_return_conditional_losses_101420

inputs
initial_state_0
initial_state_1=
)lstm_cell_3_split_readvariableop_resource:
È°	:
+lstm_cell_3_split_1_readvariableop_resource:	°	7
#lstm_cell_3_readvariableop_resource:
¬°	
identity

identity_1

identity_2¢lstm_cell_3/ReadVariableOp¢lstm_cell_3/ReadVariableOp_1¢lstm_cell_3/ReadVariableOp_2¢lstm_cell_3/ReadVariableOp_3¢ lstm_cell_3/split/ReadVariableOp¢"lstm_cell_3/split_1/ReadVariableOp¢whilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈB
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_maskc
lstm_cell_3/ones_like/ShapeShapestrided_slice_1:output:0*
T0*
_output_shapes
:`
lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_3/ones_likeFill$lstm_cell_3/ones_like/Shape:output:0$lstm_cell_3/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ^
lstm_cell_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?
lstm_cell_3/dropout/MulMullstm_cell_3/ones_like:output:0"lstm_cell_3/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈg
lstm_cell_3/dropout/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:¥
0lstm_cell_3/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_3/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0g
"lstm_cell_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>Ë
 lstm_cell_3/dropout/GreaterEqualGreaterEqual9lstm_cell_3/dropout/random_uniform/RandomUniform:output:0+lstm_cell_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/dropout/CastCast$lstm_cell_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/dropout/Mul_1Mullstm_cell_3/dropout/Mul:z:0lstm_cell_3/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ`
lstm_cell_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?
lstm_cell_3/dropout_1/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈi
lstm_cell_3/dropout_1/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:©
2lstm_cell_3/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0i
$lstm_cell_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>Ñ
"lstm_cell_3/dropout_1/GreaterEqualGreaterEqual;lstm_cell_3/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/dropout_1/CastCast&lstm_cell_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/dropout_1/Mul_1Mullstm_cell_3/dropout_1/Mul:z:0lstm_cell_3/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ`
lstm_cell_3/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?
lstm_cell_3/dropout_2/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈi
lstm_cell_3/dropout_2/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:©
2lstm_cell_3/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0i
$lstm_cell_3/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>Ñ
"lstm_cell_3/dropout_2/GreaterEqualGreaterEqual;lstm_cell_3/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/dropout_2/CastCast&lstm_cell_3/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/dropout_2/Mul_1Mullstm_cell_3/dropout_2/Mul:z:0lstm_cell_3/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ`
lstm_cell_3/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?
lstm_cell_3/dropout_3/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈi
lstm_cell_3/dropout_3/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:©
2lstm_cell_3/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0i
$lstm_cell_3/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>Ñ
"lstm_cell_3/dropout_3/GreaterEqualGreaterEqual;lstm_cell_3/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/dropout_3/CastCast&lstm_cell_3/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/dropout_3/Mul_1Mullstm_cell_3/dropout_3/Mul:z:0lstm_cell_3/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ\
lstm_cell_3/ones_like_1/ShapeShapeinitial_state_0*
T0*
_output_shapes
:b
lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¢
lstm_cell_3/ones_like_1Fill&lstm_cell_3/ones_like_1/Shape:output:0&lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
lstm_cell_3/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_3/dropout_4/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_4/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
lstm_cell_3/dropout_4/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:©
2lstm_cell_3/dropout_4/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_4/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0i
$lstm_cell_3/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ñ
"lstm_cell_3/dropout_4/GreaterEqualGreaterEqual;lstm_cell_3/dropout_4/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/dropout_4/CastCast&lstm_cell_3/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/dropout_4/Mul_1Mullstm_cell_3/dropout_4/Mul:z:0lstm_cell_3/dropout_4/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
lstm_cell_3/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_3/dropout_5/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
lstm_cell_3/dropout_5/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:©
2lstm_cell_3/dropout_5/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_5/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0i
$lstm_cell_3/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ñ
"lstm_cell_3/dropout_5/GreaterEqualGreaterEqual;lstm_cell_3/dropout_5/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/dropout_5/CastCast&lstm_cell_3/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/dropout_5/Mul_1Mullstm_cell_3/dropout_5/Mul:z:0lstm_cell_3/dropout_5/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
lstm_cell_3/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_3/dropout_6/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_6/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
lstm_cell_3/dropout_6/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:©
2lstm_cell_3/dropout_6/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_6/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0i
$lstm_cell_3/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ñ
"lstm_cell_3/dropout_6/GreaterEqualGreaterEqual;lstm_cell_3/dropout_6/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/dropout_6/CastCast&lstm_cell_3/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/dropout_6/Mul_1Mullstm_cell_3/dropout_6/Mul:z:0lstm_cell_3/dropout_6/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`
lstm_cell_3/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell_3/dropout_7/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_7/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
lstm_cell_3/dropout_7/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:©
2lstm_cell_3/dropout_7/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_7/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype0i
$lstm_cell_3/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ñ
"lstm_cell_3/dropout_7/GreaterEqualGreaterEqual;lstm_cell_3/dropout_7/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/dropout_7/CastCast&lstm_cell_3/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/dropout_7/Mul_1Mullstm_cell_3/dropout_7/Mul:z:0lstm_cell_3/dropout_7/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/mulMulstrided_slice_1:output:0lstm_cell_3/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/mul_1Mulstrided_slice_1:output:0lstm_cell_3/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/mul_2Mulstrided_slice_1:output:0lstm_cell_3/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
lstm_cell_3/mul_3Mulstrided_slice_1:output:0lstm_cell_3/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ]
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_cell_3/split/ReadVariableOpReadVariableOp)lstm_cell_3_split_readvariableop_resource* 
_output_shapes
:
È°	*
dtype0Ê
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0(lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È¬:
È¬:
È¬:
È¬*
	num_split
lstm_cell_3/MatMulMatMullstm_cell_3/mul:z:0lstm_cell_3/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/MatMul_1MatMullstm_cell_3/mul_1:z:0lstm_cell_3/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/MatMul_2MatMullstm_cell_3/mul_2:z:0lstm_cell_3/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/MatMul_3MatMullstm_cell_3/mul_3:z:0lstm_cell_3/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬_
lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"lstm_cell_3/split_1/ReadVariableOpReadVariableOp+lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:°	*
dtype0¼
lstm_cell_3/split_1Split&lstm_cell_3/split_1/split_dim:output:0*lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:¬:¬:¬:¬*
	num_split
lstm_cell_3/BiasAddBiasAddlstm_cell_3/MatMul:product:0lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/BiasAdd_1BiasAddlstm_cell_3/MatMul_1:product:0lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/BiasAdd_2BiasAddlstm_cell_3/MatMul_2:product:0lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/BiasAdd_3BiasAddlstm_cell_3/MatMul_3:product:0lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬}
lstm_cell_3/mul_4Mulinitial_state_0lstm_cell_3/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬}
lstm_cell_3/mul_5Mulinitial_state_0lstm_cell_3/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬}
lstm_cell_3/mul_6Mulinitial_state_0lstm_cell_3/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬}
lstm_cell_3/mul_7Mulinitial_state_0lstm_cell_3/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/ReadVariableOpReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0p
lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  r
!lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
lstm_cell_3/strided_sliceStridedSlice"lstm_cell_3/ReadVariableOp:value:0(lstm_cell_3/strided_slice/stack:output:0*lstm_cell_3/strided_slice/stack_1:output:0*lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
lstm_cell_3/MatMul_4MatMullstm_cell_3/mul_4:z:0"lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/addAddV2lstm_cell_3/BiasAdd:output:0lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
lstm_cell_3/SigmoidSigmoidlstm_cell_3/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/ReadVariableOp_1ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0r
!lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  t
#lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  t
#lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
lstm_cell_3/strided_slice_1StridedSlice$lstm_cell_3/ReadVariableOp_1:value:0*lstm_cell_3/strided_slice_1/stack:output:0,lstm_cell_3/strided_slice_1/stack_1:output:0,lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
lstm_cell_3/MatMul_5MatMullstm_cell_3/mul_5:z:0$lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/add_1AddV2lstm_cell_3/BiasAdd_1:output:0lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬w
lstm_cell_3/mul_8Mullstm_cell_3/Sigmoid_1:y:0initial_state_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/ReadVariableOp_2ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0r
!lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  t
#lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
lstm_cell_3/strided_slice_2StridedSlice$lstm_cell_3/ReadVariableOp_2:value:0*lstm_cell_3/strided_slice_2/stack:output:0,lstm_cell_3/strided_slice_2/stack_1:output:0,lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
lstm_cell_3/MatMul_6MatMullstm_cell_3/mul_6:z:0$lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/add_2AddV2lstm_cell_3/BiasAdd_2:output:0lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬b
lstm_cell_3/TanhTanhlstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬z
lstm_cell_3/mul_9Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬{
lstm_cell_3/add_3AddV2lstm_cell_3/mul_8:z:0lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/ReadVariableOp_3ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
¬°	*
dtype0r
!lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"      t
#lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
lstm_cell_3/strided_slice_3StridedSlice$lstm_cell_3/ReadVariableOp_3:value:0*lstm_cell_3/strided_slice_3/stack:output:0,lstm_cell_3/strided_slice_3/stack_1:output:0,lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
¬¬*

begin_mask*
end_mask
lstm_cell_3/MatMul_7MatMullstm_cell_3/mul_7:z:0$lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/add_4AddV2lstm_cell_3/BiasAdd_3:output:0lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬d
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_3/mul_10Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  ¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÷
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0initial_state_0initial_state_1strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_3_split_readvariableop_resource+lstm_cell_3_split_1_readvariableop_resource#lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_101220*
condR
while_cond_101219*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
NoOpNoOp^lstm_cell_3/ReadVariableOp^lstm_cell_3/ReadVariableOp_1^lstm_cell_3/ReadVariableOp_2^lstm_cell_3/ReadVariableOp_3!^lstm_cell_3/split/ReadVariableOp#^lstm_cell_3/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : 28
lstm_cell_3/ReadVariableOplstm_cell_3/ReadVariableOp2<
lstm_cell_3/ReadVariableOp_1lstm_cell_3/ReadVariableOp_12<
lstm_cell_3/ReadVariableOp_2lstm_cell_3/ReadVariableOp_22<
lstm_cell_3/ReadVariableOp_3lstm_cell_3/ReadVariableOp_32D
 lstm_cell_3/split/ReadVariableOp lstm_cell_3/split/ReadVariableOp2H
"lstm_cell_3/split_1/ReadVariableOp"lstm_cell_3/split_1/ReadVariableOp2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:YU
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
)
_user_specified_nameinitial_state/0:YU
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
)
_user_specified_nameinitial_state/1
Ç
ö
B__inference_model_3_layer_call_and_return_conditional_losses_99401
input_2
input_7
input_5
input_6%
embedding_1_99379:
¸dÈ 
lstm_3_99382:
È°	
lstm_3_99384:	°	 
lstm_3_99386:
¬°	*
time_distributed_99391:
¬¸d%
time_distributed_99393:	¸d
identity

identity_1

identity_2¢#embedding_1/StatefulPartitionedCall¢lstm_3/StatefulPartitionedCall¢(time_distributed/StatefulPartitionedCallö
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinput_2embedding_1_99379*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_98565ê
lstm_3/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0input_5input_6lstm_3_99382lstm_3_99384lstm_3_99386*
Tin

2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_99224¿
(time_distributed/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0time_distributed_99391time_distributed_99393*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_98535o
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  ¨
time_distributed/ReshapeReshape'lstm_3/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
IdentityIdentity1time_distributed/StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸dy

Identity_1Identity'lstm_3/StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬y

Identity_2Identity'lstm_3/StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¸
NoOpNoOp$^embedding_1/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ>¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : : 2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall:Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:UQ
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>¬
!
_user_specified_name	input_7:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
!
_user_specified_name	input_5:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
!
_user_specified_name	input_6
©

õ
A__inference_dense_layer_call_and_return_conditional_losses_101764

inputs2
matmul_readvariableop_resource:
¬¸d.
biasadd_readvariableop_resource:	¸d
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¬¸d*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸ds
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¸d*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸dW
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸da
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_defaultð
D
input_29
serving_default_input_2:0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
<
input_51
serving_default_input_5:0ÿÿÿÿÿÿÿÿÿ¬
<
input_61
serving_default_input_6:0ÿÿÿÿÿÿÿÿÿ¬
@
input_75
serving_default_input_7:0ÿÿÿÿÿÿÿÿÿ>¬;
lstm_31
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ¬=
lstm_3_11
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿ¬R
time_distributed>
StatefulPartitionedCall:2ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸dtensorflow/serving/predict:
ý
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
µ

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
Ú
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
"
_tf_keras_input_layer
°
	 layer
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
J
0
'1
(2
)3
*4
+5"
trackable_list_wrapper
J
0
'1
(2
)3
*4
+5"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ê2ç
'__inference_model_3_layer_call_fn_98841
'__inference_model_3_layer_call_fn_99425
'__inference_model_3_layer_call_fn_99449
'__inference_model_3_layer_call_fn_99345À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
×2Ô
B__inference_model_3_layer_call_and_return_conditional_losses_99710
C__inference_model_3_layer_call_and_return_conditional_losses_100099
B__inference_model_3_layer_call_and_return_conditional_losses_99373
B__inference_model_3_layer_call_and_return_conditional_losses_99401À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æBã
 __inference__wrapped_model_97934input_2input_7input_5input_6"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
1serving_default"
signature_map
*:(
¸dÈ2embedding_1/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_embedding_1_layer_call_fn_100132¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_embedding_1_layer_call_and_return_conditional_losses_100142¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø
7
state_size

'kernel
(recurrent_kernel
)bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<_random_generator
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
'0
(1
)2"
trackable_list_wrapper
5
'0
(1
)2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

?states
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
ÿ2ü
'__inference_lstm_3_layer_call_fn_100157
'__inference_lstm_3_layer_call_fn_100172
'__inference_lstm_3_layer_call_fn_100189
'__inference_lstm_3_layer_call_fn_100206Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ë2è
B__inference_lstm_3_layer_call_and_return_conditional_losses_100451
B__inference_lstm_3_layer_call_and_return_conditional_losses_100824
B__inference_lstm_3_layer_call_and_return_conditional_losses_101058
B__inference_lstm_3_layer_call_and_return_conditional_losses_101420Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
»

*kernel
+bias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses"
_tf_keras_layer
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
¬2©
1__inference_time_distributed_layer_call_fn_101429
1__inference_time_distributed_layer_call_fn_101438À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
â2ß
L__inference_time_distributed_layer_call_and_return_conditional_losses_101460
L__inference_time_distributed_layer_call_and_return_conditional_losses_101482À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
-:+
È°	2lstm_3/lstm_cell_3/kernel
7:5
¬°	2#lstm_3/lstm_cell_3/recurrent_kernel
&:$°	2lstm_3/lstm_cell_3/bias
+:)
¬¸d2time_distributed/kernel
$:"¸d2time_distributed/bias
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
äBá
$__inference_signature_wrapper_100125input_2input_5input_6input_7"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
'0
(1
)2"
trackable_list_wrapper
5
'0
(1
)2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
8	variables
9trainable_variables
:regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 2
,__inference_lstm_cell_3_layer_call_fn_101499
,__inference_lstm_cell_3_layer_call_fn_101516¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
G__inference_lstm_cell_3_layer_call_and_return_conditional_losses_101598
G__inference_lstm_cell_3_layer_call_and_return_conditional_losses_101744¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_dense_layer_call_fn_101753¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_dense_layer_call_and_return_conditional_losses_101764¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
'
 0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 __inference__wrapped_model_97934ñ')(*+³¢¯
§¢£
 
*'
input_2ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
&#
input_7ÿÿÿÿÿÿÿÿÿ>¬
"
input_5ÿÿÿÿÿÿÿÿÿ¬
"
input_6ÿÿÿÿÿÿÿÿÿ¬
ª "°ª¬
+
lstm_3!
lstm_3ÿÿÿÿÿÿÿÿÿ¬
/
lstm_3_1# 
lstm_3_1ÿÿÿÿÿÿÿÿÿ¬
L
time_distributed85
time_distributedÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d£
A__inference_dense_layer_call_and_return_conditional_losses_101764^*+0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ¬
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ¸d
 {
&__inference_dense_layer_call_fn_101753Q*+0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ¬
ª "ÿÿÿÿÿÿÿÿÿ¸d½
G__inference_embedding_1_layer_call_and_return_conditional_losses_100142r8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
,__inference_embedding_1_layer_call_fn_100132e8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
B__inference_lstm_3_layer_call_and_return_conditional_losses_100451Ó')(P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ

 
p 

 
ª "z¢w
pm
+(
0/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

0/1ÿÿÿÿÿÿÿÿÿ¬

0/2ÿÿÿÿÿÿÿÿÿ¬
 
B__inference_lstm_3_layer_call_and_return_conditional_losses_100824Ó')(P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ

 
p

 
ª "z¢w
pm
+(
0/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

0/1ÿÿÿÿÿÿÿÿÿ¬

0/2ÿÿÿÿÿÿÿÿÿ¬
 ð
B__inference_lstm_3_layer_call_and_return_conditional_losses_101058©')(¥¢¡
¢
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ

 
p 
[X
*'
initial_state/0ÿÿÿÿÿÿÿÿÿ¬
*'
initial_state/1ÿÿÿÿÿÿÿÿÿ¬
ª "z¢w
pm
+(
0/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

0/1ÿÿÿÿÿÿÿÿÿ¬

0/2ÿÿÿÿÿÿÿÿÿ¬
 ð
B__inference_lstm_3_layer_call_and_return_conditional_losses_101420©')(¥¢¡
¢
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ

 
p
[X
*'
initial_state/0ÿÿÿÿÿÿÿÿÿ¬
*'
initial_state/1ÿÿÿÿÿÿÿÿÿ¬
ª "z¢w
pm
+(
0/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

0/1ÿÿÿÿÿÿÿÿÿ¬

0/2ÿÿÿÿÿÿÿÿÿ¬
 ï
'__inference_lstm_3_layer_call_fn_100157Ã')(P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ

 
p 

 
ª "jg
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

1ÿÿÿÿÿÿÿÿÿ¬

2ÿÿÿÿÿÿÿÿÿ¬ï
'__inference_lstm_3_layer_call_fn_100172Ã')(P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ

 
p

 
ª "jg
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

1ÿÿÿÿÿÿÿÿÿ¬

2ÿÿÿÿÿÿÿÿÿ¬Å
'__inference_lstm_3_layer_call_fn_100189')(¥¢¡
¢
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ

 
p 
[X
*'
initial_state/0ÿÿÿÿÿÿÿÿÿ¬
*'
initial_state/1ÿÿÿÿÿÿÿÿÿ¬
ª "jg
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

1ÿÿÿÿÿÿÿÿÿ¬

2ÿÿÿÿÿÿÿÿÿ¬Å
'__inference_lstm_3_layer_call_fn_100206')(¥¢¡
¢
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ

 
p
[X
*'
initial_state/0ÿÿÿÿÿÿÿÿÿ¬
*'
initial_state/1ÿÿÿÿÿÿÿÿÿ¬
ª "jg
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

1ÿÿÿÿÿÿÿÿÿ¬

2ÿÿÿÿÿÿÿÿÿ¬Ð
G__inference_lstm_cell_3_layer_call_and_return_conditional_losses_101598')(¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿÈ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ¬
# 
states/1ÿÿÿÿÿÿÿÿÿ¬
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ¬
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ¬
 
0/1/1ÿÿÿÿÿÿÿÿÿ¬
 Ð
G__inference_lstm_cell_3_layer_call_and_return_conditional_losses_101744')(¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿÈ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ¬
# 
states/1ÿÿÿÿÿÿÿÿÿ¬
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ¬
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ¬
 
0/1/1ÿÿÿÿÿÿÿÿÿ¬
 ¥
,__inference_lstm_cell_3_layer_call_fn_101499ô')(¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿÈ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ¬
# 
states/1ÿÿÿÿÿÿÿÿÿ¬
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ¬
C@

1/0ÿÿÿÿÿÿÿÿÿ¬

1/1ÿÿÿÿÿÿÿÿÿ¬¥
,__inference_lstm_cell_3_layer_call_fn_101516ô')(¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿÈ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ¬
# 
states/1ÿÿÿÿÿÿÿÿÿ¬
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ¬
C@

1/0ÿÿÿÿÿÿÿÿÿ¬

1/1ÿÿÿÿÿÿÿÿÿ¬
C__inference_model_3_layer_call_and_return_conditional_losses_100099Æ')(*+¿¢»
³¢¯
¤ 
+(
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
'$
inputs/1ÿÿÿÿÿÿÿÿÿ>¬
# 
inputs/2ÿÿÿÿÿÿÿÿÿ¬
# 
inputs/3ÿÿÿÿÿÿÿÿÿ¬
p

 
ª "z¢w
pm
+(
0/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d

0/1ÿÿÿÿÿÿÿÿÿ¬

0/2ÿÿÿÿÿÿÿÿÿ¬
 
B__inference_model_3_layer_call_and_return_conditional_losses_99373Â')(*+»¢·
¯¢«
 
*'
input_2ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
&#
input_7ÿÿÿÿÿÿÿÿÿ>¬
"
input_5ÿÿÿÿÿÿÿÿÿ¬
"
input_6ÿÿÿÿÿÿÿÿÿ¬
p 

 
ª "z¢w
pm
+(
0/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d

0/1ÿÿÿÿÿÿÿÿÿ¬

0/2ÿÿÿÿÿÿÿÿÿ¬
 
B__inference_model_3_layer_call_and_return_conditional_losses_99401Â')(*+»¢·
¯¢«
 
*'
input_2ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
&#
input_7ÿÿÿÿÿÿÿÿÿ>¬
"
input_5ÿÿÿÿÿÿÿÿÿ¬
"
input_6ÿÿÿÿÿÿÿÿÿ¬
p

 
ª "z¢w
pm
+(
0/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d

0/1ÿÿÿÿÿÿÿÿÿ¬

0/2ÿÿÿÿÿÿÿÿÿ¬
 
B__inference_model_3_layer_call_and_return_conditional_losses_99710Æ')(*+¿¢»
³¢¯
¤ 
+(
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
'$
inputs/1ÿÿÿÿÿÿÿÿÿ>¬
# 
inputs/2ÿÿÿÿÿÿÿÿÿ¬
# 
inputs/3ÿÿÿÿÿÿÿÿÿ¬
p 

 
ª "z¢w
pm
+(
0/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d

0/1ÿÿÿÿÿÿÿÿÿ¬

0/2ÿÿÿÿÿÿÿÿÿ¬
 Þ
'__inference_model_3_layer_call_fn_98841²')(*+»¢·
¯¢«
 
*'
input_2ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
&#
input_7ÿÿÿÿÿÿÿÿÿ>¬
"
input_5ÿÿÿÿÿÿÿÿÿ¬
"
input_6ÿÿÿÿÿÿÿÿÿ¬
p 

 
ª "jg
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d

1ÿÿÿÿÿÿÿÿÿ¬

2ÿÿÿÿÿÿÿÿÿ¬Þ
'__inference_model_3_layer_call_fn_99345²')(*+»¢·
¯¢«
 
*'
input_2ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
&#
input_7ÿÿÿÿÿÿÿÿÿ>¬
"
input_5ÿÿÿÿÿÿÿÿÿ¬
"
input_6ÿÿÿÿÿÿÿÿÿ¬
p

 
ª "jg
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d

1ÿÿÿÿÿÿÿÿÿ¬

2ÿÿÿÿÿÿÿÿÿ¬â
'__inference_model_3_layer_call_fn_99425¶')(*+¿¢»
³¢¯
¤ 
+(
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
'$
inputs/1ÿÿÿÿÿÿÿÿÿ>¬
# 
inputs/2ÿÿÿÿÿÿÿÿÿ¬
# 
inputs/3ÿÿÿÿÿÿÿÿÿ¬
p 

 
ª "jg
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d

1ÿÿÿÿÿÿÿÿÿ¬

2ÿÿÿÿÿÿÿÿÿ¬â
'__inference_model_3_layer_call_fn_99449¶')(*+¿¢»
³¢¯
¤ 
+(
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
'$
inputs/1ÿÿÿÿÿÿÿÿÿ>¬
# 
inputs/2ÿÿÿÿÿÿÿÿÿ¬
# 
inputs/3ÿÿÿÿÿÿÿÿÿ¬
p

 
ª "jg
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d

1ÿÿÿÿÿÿÿÿÿ¬

2ÿÿÿÿÿÿÿÿÿ¬¿
$__inference_signature_wrapper_100125')(*+Ø¢Ô
¢ 
ÌªÈ
5
input_2*'
input_2ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
-
input_5"
input_5ÿÿÿÿÿÿÿÿÿ¬
-
input_6"
input_6ÿÿÿÿÿÿÿÿÿ¬
1
input_7&#
input_7ÿÿÿÿÿÿÿÿÿ>¬"°ª¬
+
lstm_3!
lstm_3ÿÿÿÿÿÿÿÿÿ¬
/
lstm_3_1# 
lstm_3_1ÿÿÿÿÿÿÿÿÿ¬
L
time_distributed85
time_distributedÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸dÑ
L__inference_time_distributed_layer_call_and_return_conditional_losses_101460*+E¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
p 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d
 Ñ
L__inference_time_distributed_layer_call_and_return_conditional_losses_101482*+E¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
p

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d
 ¨
1__inference_time_distributed_layer_call_fn_101429s*+E¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d¨
1__inference_time_distributed_layer_call_fn_101438s*+E¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸d