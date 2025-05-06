from collections import Counter
import matplotlib.pyplot as plt
import re

# 假设你有一组文本列表
texts = ["I love deep learning", "Learning is fun", "I love NLP"]

# 词频统计条形图

# 分词 + 统计词频
words = []
for text in texts:
    words += re.findall(r'\w+', text.lower())

counter = Counter(words)
most_common = counter.most_common(10)

# 可视化
words, freqs = zip(*most_common)
plt.bar(words, freqs)
plt.xticks(rotation=45)
plt.title("Top 10 Frequent Words")
plt.show()

# 用词大小表示频率，形象地展示文本主题。
from wordcloud import WordCloud

text = " ".join(texts)
wordcloud = WordCloud(width=600, height=300, background_color='white').generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud")
plt.show()

# 句子长度分布
sentence_lengths = [len(text.split()) for text in texts]

plt.hist(sentence_lengths, bins=10)
plt.xlabel("Sentence Length (words)")
plt.ylabel("Count")
plt.title("Sentence Length Distribution")
plt.show()
