from openvino.runtime import Core, PartialShape

ie = Core()
model_xml = 'openvino/TextureClassifierModel.xml'

model = ie.read_model(model=model_xml)
compiled_model = ie.compile_model(model=model,device_name='CPU')

# 查看模型信息
print("model.inputs:", model.inputs)

input_layer = model.input(0)
print("input_name:", input_layer.any_name)
print(f'input precision:{input_layer.element_type}')
print(f'input partial_shape:{input_layer.partial_shape}')
# print(f'input shape:{input_layer.shape}')

print("model.inputs:", model.outputs)

output_layer = model.output(0)
print("output_name:", output_layer.any_name)
print(f'output precision:{output_layer.element_type}')
print(f'output partial_shape:{output_layer.partial_shape}')
# print(f'output shape:{output_layer.shape}')