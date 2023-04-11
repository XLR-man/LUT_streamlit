# import gradio as gr
# import evaluation_LOL
# import image_adaptive_lut_evaluation
#
# # 传入三个参数，必须以关键字参数的形式传入参数
# # 第一个参数是训练函数的名字，第二个参数是训练函数的输入类型，
# # 第三个参数是训练函数的输出类型
# interface=gr.Interface(fn=image_adaptive_lut_evaluation.functionForGradio,inputs='image',outputs='image')
# # # 如果不需要共享链接可以不用这条语句
# # interface.launch(share=True)
# # 在这里要注意一下
# # 如果指明主机地址是 0.0.0.0，就表示这个应用可以被外部访问
# # 如果主机地址是 127.0.0.1，就表示只能被本机应用访问
# # gradio 默认的端口是 7860，可是当 7860 被占用时它会使用其他端口
# # 如果指明了端口的话，如果端口被占用应用就不会启动，可以起到提示作用
# interface.launch(server_name="0.0.0.0",server_port=6006,share=True)

import gradio as gr
import cv2

def to_black(image):
    output = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return output

interface = gr.Interface(fn=to_black, inputs="image", outputs="image")
interface.launch(server_name="0.0.0.0",server_port=6006,share=True)