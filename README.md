# YouTube Chatbot 🤖

YouTubeの動画内容を理解し、動画に関する質問に答えたり、特定のキーワードがいつ話されているかを教えてくれるAIチャットボットです。

## 🌟 主な機能

### 1. **動画内容の質問応答**
- YouTube動画の内容について質問すると、動画の字幕を基に回答を生成
- 要約、詳細な説明、特定の話題について質問可能

### 2. **キーワード時間検索**
- 特定のキーワードやトピックが動画のどの時間帯で話されているかを検索
- 長い動画から必要な部分を素早く見つけることが可能

### 3. **インタラクティブなチャットUI**
- Streamlitベースの使いやすいチャットインターフェース
- チャット履歴の保持
- リアルタイムでの応答

## 🚀 技術スタック

- **Streamlit**: Webアプリケーションフレームワーク
- **LangChain**: LLMとの連携とプロンプト管理
- **OpenAI GPT-3.5**: 自然言語処理エンジン
- **YouTube Transcript API**: YouTube字幕データの取得
- **Python 3.8+**: 開発言語

## 📦 インストールと初期設定

### 1. リポジトリのクローン
```bash
git clone https://github.com/dumbled0re/youtube-chatbot.git
cd youtube-chatbot
```

### 2. 依存関係のインストール

#### Poetryを使用する場合（推奨）
```bash
poetry install
poetry shell
```

#### pipを使用する場合
```bash
pip install streamlit openai langchain youtube-transcript-api python-dotenv
```

### 3. 環境変数の設定
`.env`ファイルを作成し、OpenAI APIキーを設定してください：

```bash
touch .env
```

`.env`ファイルを編集：
```
OPENAI_API_KEY=your_openai_api_key_here
```

**OpenAI APIキーの取得方法:**
1. [OpenAI](https://platform.openai.com/)にアクセス
2. アカウントを作成またはログイン
3. API Keys セクションで新しいAPIキーを生成

## 🖥️ 使い方

### アプリケーションの起動
```bash
streamlit run main.py
```

ブラウザで `http://localhost:8501` が自動的に開きます。

### 基本的な使い方

1. **YouTube URLの入力**
   - YouTube動画のURLを入力欄に貼り付け
   - 日本語字幕が利用可能な動画を推奨

2. **質問の例**
   
   **動画内容に関する質問:**
   - 「この動画を要約して」
   - 「主なポイントを教えて」
   - 「〇〇について詳しく説明して」
   
   **時間検索:**
   - 「機械学習について」
   - 「デモンストレーションの部分」
   - 「質疑応答の時間」

3. **結果の表示**
   - 内容質問：動画の内容を基にした詳細な回答
   - 時間検索：「〇〇の説明は動画の X分Y秒からZ分W秒です」

## 💡 使用例

### 例1: 動画の要約
```
ユーザー: 「この動画の内容を要約してください」
ボット: 「この動画では、機械学習の基本概念について説明されています。
主な内容は以下の通りです：
1. 教師あり学習の概念
2. データ前処理の重要性
3. モデルの評価方法
...」
```

### 例2: 特定の話題の時間検索
```
ユーザー: 「ニューラルネットワークについて」
ボット: 「ニューラルネットワークの説明は動画の5分30秒から8分15秒です。」
```

## 🔧 カスタマイズ

### パラメータの調整
`main.py`内で以下の設定を変更できます：

- **分割時間**: `split_duration=60` (デフォルト60秒)
- **使用モデル**: `model_name="gpt-3.5-turbo-16k"`
- **言語設定**: `language=['ja']` (日本語)

### 新機能の追加
`functions`配列に新しいfunction callingの定義を追加することで、機能を拡張できます。

## 📁 プロジェクト構造

```
youtube-chatbot/
├── main.py              # メインアプリケーション
├── chatbot.py           # チャットボット機能
├── dalle3.py            # DALL-E 3関連機能
├── pyproject.toml       # Poetry設定ファイル
├── poetry.lock          # Poetry依存関係ロック
├── .flake8             # Flake8設定
├── .gitignore          # Git除外ファイル
└── README.md           # このファイル
```

## 🔍 主要な関数

- `generate_video_response()`: 動画内容に関する質問への回答生成
- `generate_video_time_response()`: キーワードの時間検索
- `split_text_by_time_intervals()`: 字幕の時間分割
- `call_chatbot_function()`: Function Callingによる機能振り分け

## 🐛 トラブルシューティング

### よくある問題と解決方法

1. **字幕が取得できない**
   - 動画に日本語字幕があることを確認
   - 動画が公開状態であることを確認

2. **OpenAI APIエラー**
   - APIキーが正しく設定されているか確認
   - APIクォータが残っているか確認

3. **依存関係エラー**
   ```bash
   poetry install --no-dev
   # または
   pip install --upgrade streamlit openai langchain youtube-transcript-api python-dotenv
   ```

## 🎯 対応する動画の条件

- **字幕が利用可能**: 日本語の字幕または自動字幕が必要
- **公開動画**: プライベート動画は対応していません
- **長さ制限**: 特に制限はありませんが、非常に長い動画の場合は処理時間が長くなる可能性があります

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🤝 コントリビューション

バグ報告、機能要求、プルリクエストを歓迎します！

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチをプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📞 サポート

質問や問題がある場合は、GitHubのIssuesページでお気軽にお問い合わせください。

---

**注意**: このツールはYouTubeの利用規約に従って使用してください。教育目的や個人的な学習のための利用を想定しています。