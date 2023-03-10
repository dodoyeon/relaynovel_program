from flask import Flask, render_template, request
from flask_restx import Resource, Api
import requests

import json


# Hompage SERVER
app = Flask(__name__)
novel = []
count = 0

# post, get 같은 rest 함수들의 이름이 미리 정해진 상태에서 덮어씌우는 식으로 선언하는거라 이름이 바뀌면 작동안한다
@app.route('/', methods=['GET', 'POST'])
def post():
    if request.method == 'POST':
        global novel
        global count
        
        # request.form[''] 은 html에서 form을 통해서 보낸 정보를 받는 함수
        sentence = request.form['input'] 
        # print(sentence)
        novel.append(['U', sentence])

        # url 주소에 어떤 함수에게 POST 요청과 함께 json=data 데이터를 보내고
        # 그쪽에서 처리하고 return된 아웃풋을 다시 받아옴

        # data = {'sentence': sentence}
        # result = requests.post([url], json=data)
        # result = result.json()
        # print(result)
        # response = result['response']

        # response = generator(sentence)

        response = 'Test Inference Success!'
        # novel += response -> string은 char의 list이므로 +가 아니라 .append() 사용해야한다.
        novel.append(['G', response])

        count += 2
        # relay = {
        #     'count' : count,
        #     'sentence': sentence,
        #     'result': response,
        #     'novel': novel
        # }

    return render_template('layout.html', inference_log=novel)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
