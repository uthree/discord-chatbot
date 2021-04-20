import os
import yaml
import uuid
import asyncio
import traceback

from discord.ext import commands
import discord as discord

default_token_file = {
    'using': 'main',
    "main": '<YOUR BOT TOKEN HERE>'
}

class ChatBot(commands.Bot):
    def __init__(self, command_prefix):
        super().__init__(command_prefix)
        # INITIAL_COGSに格納されている名前から、コグを読み込む。
        # エラーが発生した場合は、エラー内容を表示。
        for cog in INITIAL_EXTENSIONS:
            try:
                self.load_extension(cog)
            except Exception:
                traceback.print_exc()
    
    async def on_message(self, message):
        # 送信者が自分自身であれば反応しない。
        if message.author.id == (await self.application_info()).id:
            return

        if type(message.channel) == discord.DMChannel:
            await self.on_direct_message(message)

    async def on_direct_message(self, message):
        await message.channel.send("fuck you")


def main():
    if not os.path.exists("token.yml"):
        # トークンファイルがない場合は自動的に作成。
        with open("token.yml", "w") as file:
            yaml.dump(default_token_file, file)
            print(
                """
                token.ymlファイルを編集して、botトークンを追加してください。
                デフォルトでは main: 項目に追加することでbotが使用可能になります。
                """
            )
            exit()
    else:
        # トークン読み込み処理
        with open("token.yml") as file:
            token_data = yaml.safe_load(file)
        using_token = token_data['using']
        token = token_data[using_token]
        print(f"{using_token} のトークンを使用します。")

        # cog読み込み処理
        with open("load_cogs.yml") as file:
            global INITIAL_EXTENSIONS
            INITIAL_EXTENSIONS = yaml.safe_load(file)['cogs']
        # 内部的なprefixをuuidにしてわかりにくくする処理
        uuidpref = str(uuid.uuid4())
        print(f"prefix: {uuidpref}")

        # Botのインスタンス化及び起動処理。
        bot = ChatBot(command_prefix=uuidpref)
        bot.run(token)  # Botのトークンを入れて実行

if __name__ == "__main__":
    main()