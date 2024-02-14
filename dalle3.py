import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


client = OpenAI(
    organization="",
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# プロンプト・画像サイズ・画像の品質・生成回数を指定
response = client.images.generate(
    model="dall-e-3",
    prompt="ツイッターのアイコンにふさわしい男性を生成してください。",
    size="1024x1024",
    quality="standard",
    n=1,
)

image_url = response.data[0].url

print(image_url)
