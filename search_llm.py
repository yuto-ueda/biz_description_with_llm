import json
import os

import dotenv
import openai
import promptlayer
import regex
from langchain.chains import LLMChain, LLMRequestsChain, SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.promptlayer_openai import PromptLayerChatOpenAI
from langchain.prompts import PromptTemplate

dotenv.load_dotenv()

promptlayer.api_key = os.environ["PROMPTLAYER_API_KEY"]
openai = promptlayer.openai


def extract_json_strings(text):
    json_pattern = r'\{(?:[^{}]|(?R))*\}'
    js = regex.findall(json_pattern, text)[0]
    return json.loads(js)


def get_biz_description(company_name):

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0)

    # llm = PromptLayerChatOpenAI(
    #     model_name="gpt-3.5-turbo",
    #     temperature=0)


    template = """>>>と<<<の間がgoogleの検索結果です。
検索結果を用いて「{query}」という質問に対する答えを考えてください。
情報が含まれていない場合は「not found」と表示します。
>>> {requests_result}<<<"""

    PROMPT = PromptTemplate(
        input_variables=["query", "requests_result"],
        template=template,
    )

    chain = LLMRequestsChain(
        llm_chain=LLMChain(
            llm=llm,
            prompt=PROMPT
        ),
        verbose=True
    )

    question = f"{company_name}が行っている事業の概要は？"
    inputs = {
        "query": question,
        "url": "https://www.google.com/search?q=" + question.replace(" ", "+")
    }

    return chain(inputs)["output"]


def get_biz_category(biz_description):
    prompt = f"""## 事業概要
{biz_description}


## 指示
上記の「事業内容」を行っている企業について、以下のどの分類に当てはまるかを考えてください。
回答は次のような完全なjson形式で行ってください。
{{"Type": Int,
"Customer": Int,
"Positioning": Int,
"Business" : Int}}

## 分類
Type
1: バーティカル: 特定産業に特化してサービスを提供
2: ホリゾンタル: 産業を横断して特定の機能を提供
3: その他（不明な場合はその他を選んでください）

Customer
1. BtoB: 法人向け
2. BtoC: 消費者向け

Positioning
1: プラットフォーマー
2: プレイヤー
3: 受託開発・開発支援
4. その他（不明な場合はその他を選んでください）

Business
1. 情報メディア
2. 送客メディア
3. SNS
4. ゲーム、エンタメ
5. SaaS
6. EC
7. データ分析
8. その他（不明な場合はその他を選んでください）"""

    messages = [
        {"role": "user", "content":prompt}
    ]
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature=0,
    messages=messages,
    pl_tags=["get biz category"],
    )

    response_message = response["choices"][0]["message"]["content"]

    return extract_json_strings(response_message)


if __name__=="__main__":
    company_name = "株式会社ユーザベース"
    biz_description = get_biz_description(company_name)
    print(biz_description)
    biz_category = get_biz_category(biz_description)
    print(biz_category)