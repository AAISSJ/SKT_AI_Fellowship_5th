import os
import openai
import pandas as pd
import numpy as np
import json
import requests

import argparse
import itertools
from collections import Counter

from langchain import PromptTemplate
from langchain import LLMChain, OpenAI
from langchain.llms import Writer
from langchain.memory import ConversationTokenBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate


import matplotlib.pyplot as plt
from imageio import imread
from PIL import Image
import PIL

import urllib
from urllib.request import Request, urlopen
import googletrans # googletrans==4.0.0rc1


os.environ['OPENAI_API_KEY'] = "your_api"
os.environ['STABLE_DIFFUSION_API_KEY'] = "your_api"

openai.api_key = os.environ['OPENAI_API_KEY']


def make_kg_module(df,topic_df,customer_df, module_num):
    m_df = df[df['modularity_class']==module_num] # 모듈 번호 수정


    # 고객 & 상품 & 토픽 index
    m_customer = m_df[m_df['d0'] == 'customer']['Id'].to_list()
    m_customer = [int(s.replace('C_', '')) for s in m_customer]

    m_product = m_df[m_df['d0'] == 'product']['Id'].to_list()
    m_product = [int(s.replace('P_', '')) for s in m_product]

    m_topic = m_df[m_df['d0'] == 'topic']['Id'].to_list()
    m_topic = [int(s.replace('T_', '')) for s in m_topic]


    # m_customer_df = customer_df[customer_df['cust_num_graph'].isin(m_customer)]

    # 모듈 내 토픽 확인
    categories_1 = list(set(topic_df[topic_df['topic_id_graph'].isin(m_topic)]['cat1_kor_nm']))
    categories_2 = list(set(topic_df[topic_df['topic_id_graph'].isin(m_topic)]['cat2_kor_nm']))


    ####### 토픽 지식 그래프 생성
    combinations = list(itertools.combinations(categories_1, 2))

    # 각 조합을 ('값1', '관계', '값2') 형태로 변환
    topic_graph = [str((comb[0], 'topic', comb[1])) for comb in combinations]
    topic_graph_str = ' \n '.join(topic_graph)

    belong_graph = (topic_df[topic_df['topic_id_graph'].isin(m_topic)].apply(lambda row: str((row['cat1_kor_nm'], 'belong', row['cat2_kor_nm'])), axis=1)).to_list()
    belong_graph_str = ' \n '.join(belong_graph)



    ####### 타겟 고객
    m_customer_df = customer_df[customer_df['cust_num_graph'].isin(m_customer)]
    count_dict = dict(Counter(m_customer_df['seg_desc']))
    sorted_dict = dict(sorted(count_dict.items(), key=lambda item: item[1], reverse=True))
    sorted_df = pd.DataFrame(list(sorted_dict.items()), columns=['Seg_desc', 'Count'])

    return topic_graph_str, belong_graph_str, sorted_df


def filter_rows(df, age_group=None, pay_cnt_all=None, mobile_usage_all=None, pay_cnt_age_group=None, mobile_usage_age_group=None):
    mask = [True] * len(df)

    if age_group is not None:
        mask &= df['age_group'] == age_group

    if pay_cnt_all is not None:
        mask &= df['pay_cnt_all'] == pay_cnt_all

    if mobile_usage_all is not None:
        mask &= df['mobile_usage_all'] == mobile_usage_all

    if pay_cnt_age_group is not None:
        mask &= df['pay_cnt_age_group'] == pay_cnt_age_group

    if mobile_usage_age_group is not None:
        mask &= df['mobile_usage_age_group'] == mobile_usage_age_group

    return df[mask]


def read_static_elements():
    with open('prompt_text/title.txt') as file:
        title = file.read()
    
    with open('prompt_text/per_role.txt') as file:
        per_role = file.read()    
    
    with open('prompt_text/persona.txt') as file:
        persona = file.read()    
    
    with open('prompt_text/role.txt') as file:
        role = file.read()    
    
    with open('prompt_text/product.txt') as file:
        product = file.read()    
    
    with open('prompt_text/task.txt') as file:
        task = file.read()   

    return title,per_role, persona,role, product, task


def make_segment_desc(df):
    
    segment_desc = ''
    
    for index, row in df.iterrows():
        row_str = ', '.join(row[['age_group','seg_nm_kor','seg_desc','seg_keyword']].map(str))
        segment_desc+=f'Segment {index}: {row_str} \n '
    
    return segment_desc

    
def prompting_with_LangChain(title,per_role,persona,role,segment, product, task, Check, with_graph,topic_graph=None):
    
    llm = OpenAI(model_name="gpt-4")
    memory = ConversationTokenBufferMemory(llm=llm, human_prefix="동료 카피라이터", ai_prefix="토피")
    
    
    ### formatting 
    
    template = """ Title: {title}
    """

    prompt = PromptTemplate(
        input_variables=["title"],
        template=template
    )


    formatted_prompt = prompt.format(
        title = title
        )

    print(formatted_prompt)
    
            
    template = formatted_prompt+ """

    이전 대화:
    {history}

    동료 카피라이터: {input}
    토피:
    """

    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
    conversation = ConversationChain(
        prompt=PROMPT,
        llm=llm,
        verbose=True,
        memory=memory
    )


    conversation.run(input = per_role) 
    # conversation.run(input = persona)
    # conversation.run(input = role)
    conversation.run(input = segment)
    conversation.run(input = product)
    conversation.run(input = task)
    if with_graph:
        conversation.run(input = topic_graph)
    final_result = conversation.run(input = Check)
    
    return final_result


def make_image_with_Dalle(my_prompt,num_pic, size):
    
    response = openai.Image.create(
        prompt =my_prompt,
        n = num_pic,
        size = f"{size}x{size}"
    )
    
    for i in range(num_pic):
        image_url = response['data'][i]['url']
        urllib.request.urlretrieve(image_url, f'output_image/dalle_{str(i)}.png')

        # img = plt.imread(urllib.request.urlretrieve(image_url)[0])
        # plt.imshow(img)


def make_image_with_Stable(my_prompt,num_pic, size): 
    
    stable_api_key = os.environ['STABLE_DIFFUSION_API_KEY']
    url = "https://stablediffusionapi.com/api/v3/text2img"

    payload = json.dumps({
    "key": stable_api_key,
    "prompt": my_prompt,
    "negative_prompt": None,
    "width": size,
    "height": size,
    "samples": num_pic,
    "num_inference_steps": "20",
    "seed": None,
    "guidance_scale": 7.5,
    "safety_checker": "yes",
    "multi_lingual": "no",
    "panorama": "no",
    "self_attention": "no",
    "upscale": "no",
    "webhook": None,
    "track_id": None
    })

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    json_object = json.loads(response.text)
    json_object
    
    try : 
        for i in range(num_pic):
            image_url = json_object['output'][i]

            req = Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req) as u:
                raw_data = u.read()

            img = Image.open(io.BytesIO(raw_data))
            # plt.imshow(img)
            
    except:
        print("___")



def main(config):
    
    path = config['data_path']
    num_msg = config['num_msg']
    with_graph = config['with_graph']
    seg_num = config['seg_num']
    
    
    # segment 
    age_group = config['age_group'] # '청소년'
    pay_cnt_all= config['pay_cnt'] # None # '하'
    mobile_usage_all = config['mobile_usage'] # None# '하'
    pay_cnt_age_group = config['pay_cnt_age_group'] # None #'하'
    mobile_usage_age_group = config['mobile_usage_age_group' ]# None # ''
    
    # image 
    size = config['image_size'] # 256 / 512
    num_pic = config['num_pic'] # 5
    style = config['image_style']
    
    
    print("___data load___")
    
    topic_df = pd.read_json(path+'topic.json')
    tdeal_df = pd.read_json(path + 'tdeal.json')
    segment_df = pd.read_csv(path+'seg_meta.csv')
    module_df = pd.read_csv(path+'modularity_9.csv')
    customer_df = pd.read_csv(path+'warmstart_customer_6670.csv')


    print("__static elements__ ")
    title, per_role, persona,role, product, task = read_static_elements()

    
    print("__elements__")
        
    
    #### segment    
    if not with_graph: 
        filtered_df = filter_rows(segment_df, age_group=age_group, pay_cnt_all= pay_cnt_all, mobile_usage_all=mobile_usage_all,pay_cnt_age_group=pay_cnt_age_group,mobile_usage_age_group=mobile_usage_age_group )
        segment_desc = make_segment_desc(filtered_df)
    else : 
        topic_graph, belong_graph, sorted_df = make_kg_module(module_df,topic_df,customer_df, seg_num)
        segment_desc = sorted_df['Seg_desc'][0] 
    
    
    segment = f"""
        타겟 고객의 정보: {segment_desc}
        -----
        조건 1) 타겟 고객의 정보를 고려하여 마케팅 메세지를 생성한다.

        -----
        이해했으면 해당 타겟 고객군에 대한 요약을 알려줘
    """
    
    
    ### topic graph 
    if with_graph : 
        topic_graph = f"""
            그래프:
            - 지식 그래프의 노드는 토픽 키워드를 의미한다.
            - 두 노드는 topic 엣지와 belong 엣지로 연결되어 있다.
            - topic 엣지는 모든 대분류 토픽 키워드 간에 연결되며, belong 엣지는 종속 관계가 있는데 대분류 토픽과 소분류 토픽을 연결한다.
            - 아래 지식 그래프는 타겟 고객의 관심사 토픽 그래프이다. {str(topic_graph)}
            - 아래 지식 그래프는 타겟 고객의 추가된 관심사 토픽 그래프이다.  {str(belong_graph)}
            -----
            지금 입력하는 [지식 그래프]도 같이 고려하고, 다시 생성한 마케팅 문구를 알려줘.
        """
    else : 
        topic_graph = None


    ### check 
    Check = f"""
        올바른 답을 가지고 있는지 확인하기 위해 단계적으로 지시를 해결한다.
        - 핵심 내용만 간추린 간단 명료한 메세지를 선호한다.
        - 하나의 메세지에는 하나의 정보만 넣는다.
        - 짧고 간결한 문장을 만든다.
        - 자극적인 표현과 불필요한 수식어를 지양한다.
        - 고객을 존중하는 문장을 사용한다.
        - 강조하는 문장 부호는 필요한 부분에 하나만 넣는다.
        -----
        검토 사항에 따라 마케팅 메세지 {str(num_msg)}개를 만들어줘.
        최종적으로 반환되는 형식은

        - 제목: / 본문:
        - 제목: / 본문:
        - ...

        이어야 한다.

    """
    
    final_result = prompting_with_LangChain(title,per_role,persona,role,segment, product, task, Check,with_graph,topic_graph)

    print(final_result)
    
    
    
    translator = googletrans.Translator()
    result = translator.translate(final_result, dest='en')
    
    
    
    image_prompt = f"""
        Please create an image that fits the title and body below
        Do not include any text

        style: {style}
        {result.text}
    """

    make_image_with_Dalle(image_prompt[:1000],num_pic, size)     
    # openai.error.InvalidRequestError: Prompt must be length 1000 or less. Your prompt length is 1157. Please reduce your prompt length.
    
    make_image_with_Stable(image_prompt,num_pic, size)
    
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser("main.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # settings 
    parser.add_argument("--data_path", type=str, default='./data/')
    parser.add_argument("--with_graph", default=False)
    parser.add_argument("--num_msg", type=int, default=3)
    parser.add_argument("--seg_num", type=int, default=0) # 0~8 

    # segment 
    parser.add_argument("--age_group", default='청년')
    parser.add_argument("--pay_cnt", default=None) #'하' #'중' #'상'
    parser.add_argument("--mobile_usage", default=None)
    parser.add_argument("--pay_cnt_age_group", default=None)
    parser.add_argument("--mobile_usage_age_group", default=None)
    
    # image settings
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--num_pic", type=int, default=3) # 최대 4개 
    parser.add_argument("--image_style", type=str, default="Cartoon")  

    config = parser.parse_args()
    print(config)
    
    main(config.__dict__)

