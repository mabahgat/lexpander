from agents import HugChat, HuggingChatApiHelper
from config import global_config

liwc_2015_labels = [l.lower() for l in
					['affect', 'social', 'cogproc', 'percept', 'bio', 'drives', 'relativ', 'pconcern', 'informal']]


def test_hugchat_answer():
	chat = HugChat('test', ['test'])
	assert chat.answer('test') is not None


def test_hugchat_extract_label():
	chat = HugChat('LIWC 2015', liwc_2015_labels)
	label = chat.label('my brother was sick last night', 'brother')
	assert label in liwc_2015_labels


def test_hugging_chat_api_new_conversation():
	chat = HuggingChatApiHelper(global_config.apis.hugchat.user_id, global_config.apis.hugchat.token)
	chat.new_conversation()
	assert chat.current_conversation_id is not None
	chat.delete_conversation()


def test_hugging_chat_api_ask():
	chat = HuggingChatApiHelper(global_config.apis.hugchat.user_id, global_config.apis.hugchat.token)
	chat.new_conversation()
	try:
		response = chat.chat('hi')
	finally:
		chat.delete_conversation()
	assert type(response) == str
