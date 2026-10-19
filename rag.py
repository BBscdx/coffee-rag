import os
import math
import jieba
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv


# 1. 设定知识库文档所在的文件夹路径
docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'knowledge_base')
chunks = []

# 2. 遍历文件夹下的所有文件
for fname in sorted(os.listdir(docs_dir)):
    if not fname.endswith('.md'):
        continue

    file_path = os.path.join(docs_dir, fname)
    text = open(file_path, encoding='utf-8').read()

    # 3. 提取文档级来源：放宽匹配，以防原格式有变
    doc_source = ''     #这个 doc_source 是兜底来源：如果某个块的标题里没有单独写 （来源：...），就用这个文档级来源。
    lines = text.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('- 来源：'):
            if line.strip() == '- 来源：':  #如果是裸行，往下收集缩进的行
                more = []  # 先把缩进行收集进列表，最后用分号拼接（比空格好读）
                for j in range(i + 1 , len(lines)):
                    next_line = lines[j]
                    # 以空格或 tab 开头，且非空行，就是缩进的 URL
                    if (next_line.startswith(' ') or next_line.startswith('\t')) and next_line.strip():
                        more.append(next_line.strip().lstrip('- '))  # 去掉开头的 '- ' 子弹头
                    else:
                        break  # 遇到非缩进行（如 - 发布日期：）停止收集
                doc_source = '- 来源：' + '；'.join(more)
            else:
                  # 如果本身就带有 URL，直接取
                doc_source = line.strip()
            break

    # 4. 【显式切分】
    # 用 '\n## ' 切小节，避免误切 '### '
    sections = text.split('\n## ')
    
    for sec in sections[1:]:  # 第0段是文档大标题及元数据，丢弃。第一个##之前的内容。
        # 提取当前 ## 层的标题和正文
        lines = sec.split('\n', 1)   #只切一次换行
        title = lines[0].strip()  #取标题，去掉首尾空白
        body = lines[1] if len(lines) > 1 else ''   #如果存在正文就取，否则给空字符串，防止后面报错。

        # 检查当前小节内部是否有 '### ' 三级标题
        sub_sections = body.split('\n### ')
        '''如果正文里没有三级标题，sub_sections 长度为 1，sub_sections[0] 就是整个正文。
        如果正文里有三级标题，sub_sections[0] 是二级标题下、第一个三级标题之前的内容；后面的每个元素就是三级标题块'''
       
        # --- 处理 ## 当前块 ---
        body_main = sub_sections[0]   #二级标题下、第一个三级标题之前的正文。
        # 【过滤垃圾块】去掉空白后不足20字符的跳过
        if len(body_main.strip()) >= 20:
            # 【来源按块提取】
            source = doc_source  # 默认退回文档级
            #1. 优先匹配带有（来源：）的标题
            if '（来源：' in title:
                source = title.split('（来源：', 1)[1]
                if source.endswith('）'):
                    source = source[:-1]  # 剥掉结尾的 '）'
            # 2. 匹配 07 的 "来源1｜xxx" 格式（注意：是全角竖线 ｜，不是半角 |）
            elif title.startswith('来源') and '｜' in title:
                source = title.split('｜', 1)[1].strip()
            # 3. 匹配 07/08 的 "补充来源xxx" 格式
            elif title.startswith('补充来源'):
                source = title
                
            chunks.append({
                "doc_name": fname,
                "source": source,
                "title": title,
                "body": body_main.strip()
            })
            

        # --- 处理 ### 子块（如果有） ---
        for sub_sec in sub_sections[1:]:
            sub_lines = sub_sec.split('\n', 1)
            sub_title = sub_lines[0].strip()
            sub_body = sub_lines[1] if len(sub_lines) > 1 else ''

            # 过滤垃圾块
            if len(sub_body.strip()) >= 20:
                # 来源按块提取：和上面 ## 层用同样的三条规则（子块也要识别 来源N｜ 和 补充来源）
                source = doc_source
                if '（来源：' in sub_title:
                    source = sub_title.split('（来源：', 1)[1]
                    if source.endswith('）'):
                        source = source[:-1]
                # 匹配 "来源1｜xxx" 格式（全角竖线）
                elif sub_title.startswith('来源') and '｜' in sub_title:
                    source = sub_title.split('｜', 1)[1].strip()
                # 匹配 "补充来源xxx" 格式
                elif sub_title.startswith('补充来源'):
                    source = sub_title
                
                chunks.append({
                    "doc_name": fname,
                    "source": source,
                    "title": sub_title,
                    "body": sub_body.strip()
                })



# 1. 分词函数
def tokenize(text):
    # jieba.lcut 精确模式分词，并过滤掉纯空白字符
    return [word for word in jieba.lcut(text) if word.strip()]

# 2. 建索引函数
def build_index(chunks):
    N = len(chunks)
    df = {}  #记录每个词出现在多少个文档中
    tokenized_bodies = [] #缓存所有文档的分词结果，避免后续重复分词
    vocab = set() #词表，记录所有出现过的唯一词

    # 第一步：遍历所有块，统计词频和文档频率 (DF)
    for chunk in chunks:
        tokens = tokenize(chunk['body'])
        tokenized_bodies.append(tokens)
        
        # 对当前块的词去重，用于统计 DF
        unique_tokens = set(tokens)  #set(tokens) 去重，因为 DF 统计的是“出现该词的文档数”，同一个文档内出现多次只算一次
        for word in unique_tokens:
            df[word] = df.get(word, 0) + 1
            vocab.add(word)

    # 第二步：计算 IDF
    # 公式：idf(w) = log((N+1) / (df(w)+1)) + 1。加 1 是为了防止 df 为 0 或者 df=N 时导致 log 结果为 0 或负数，保持 IDF 的基本权重
    idf = {}
    for word in vocab:
        idf[word] = math.log((N + 1) / (df[word] + 1)) + 1

    # 第三步：构建每个块的 TF-IDF 向量
    doc_vectors = []
    for tokens in tokenized_bodies:
        tf = Counter(tokens)  #使用 Counter 统计每个块中每个词的出现次数     
        doc_len = len(tokens)
        # 公式：tf(w,d) = 词w在块d出现次数 / 块d总词数
        # 向量：tf * idf
        vec = {word: (tf[word] / doc_len) * idf[word] for word in tf}
        doc_vectors.append(vec)

    return doc_vectors, idf, N


# 4. 初始化索引并运行检查点
doc_vectors, idf, N = build_index(chunks)

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))   # 锚定 rag.py 所在目录，不依赖启动位置
client = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url='https://api.deepseek.com')

SYSTEM_PROMPT ="""
# 角色设定
你是一个严谨的“云南与贵阳咖啡产业”知识库问答助手。你的唯一任务是基于系统提供的参考资料，准确、客观地回答用户关于云南与贵阳咖啡产业的问题。**当前知识库仅收录 2023-2026 年的 11 篇产业资料（涵盖官方答复、产业报告、海关数据、调研纪要、新闻报道），每篇均标注来源与日期。**

# 核心规则（四道反幻觉防线，请严格遵守）
1. **绝对忠实于资料**：你只能使用检索注入的参考资料来回答问题。绝对禁止使用你训练记忆中关于云南与贵阳咖啡的任何数字、事实或判断。
2. **强制标来源**：你回答中的每一个数据、事实或结论，都必须注明出处。请在相关语句末尾标注对应的资料编号，格式如 `[资料1]`、`[资料2]`。
3. **拒绝编造**：如果参考资料里找不到回答用户问题所需的信息，你必须如实回答“资料中没有相关信息”，严禁推测、编造或拼凑答案。
4. **口径差异必须说明**：如果同一项数据在资料中出现了不同的统计口径（例如：农业产值 86.6 亿元与全产业链产值 995 亿元），你必须将不同口径的数据全部列出，并解释它们统计范围的区别，绝对不允许只挑其中一个口径回答。
5. **数字纪律**：引用数字必须与资料原文逐字一致，不得四舍五入、改写单位，不得自行添加或删除“约”“达”“超”等修饰词。资料写“约17万吨”就写“约17万吨”，资料写“17.07万吨”就写“17.07万吨”，不可混淆。

# 回答风格
保持专业、客观、简洁。回答时优先陈述事实，如果资料间的数据存在冲突或口径不同，需按规则4进行处理。"""

def search(question, top_k=5):
    q_tokens = tokenize(question)
    q_tf = Counter(q_tokens)
    q_len = len(q_tokens)
    
    q_vec = {}
    for word in q_tf:
        if word in idf: # 过滤未登录词，修复 Q1 分数稀释 Bug
            q_vec[word] = (q_tf[word] / q_len) * idf[word]
        
    q_norm = math.sqrt(sum(v ** 2 for v in q_vec.values()))
    if q_norm == 0:
        print(f"查询: '{question}' -> 没有任何有效词汇能匹配到知识库。")
        return []

    results = []
    for i, doc_vec in enumerate(doc_vectors):
        dot_product = sum(q_vec.get(w, 0) * doc_vec.get(w, 0) for w in q_vec)
        doc_norm = math.sqrt(sum(v ** 2 for v in doc_vec.values()))
        sim = dot_product / (q_norm * doc_norm + 1e-8)
        results.append((sim, i))

      
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]


def build_prompt(question, results):
    prompt_parts = []
    prompt_parts.append("# 检索到的参考资料")
    prompt_parts.append("以下是系统检索到的相关知识块，请严格依据它们回答问题。\n")
    
    for i, (sim, idx) in enumerate(results, 1):
        chunk = chunks[idx]
        prompt_parts.append(f"[资料{i}]")
        prompt_parts.append(f"标题：{chunk['title']}")
        prompt_parts.append(f"来源：{chunk['source']}")
        prompt_parts.append(f"正文：{chunk['body']}\n") # 全文注入，不截断
        
    prompt_parts.append("# 用户提问")
    prompt_parts.append(question)
    
    return "\n".join(prompt_parts)

def ask(question, top_k=5):
    print(f"\n🔍 查询: '{question}'")
    
    # 1. 检索
    top_results = search(question, top_k)
    if not top_results:
        print("❌ 检索不到相关资料，直接拒答。")
        return None, "no_match"
        
    # 2. 打印检索留痕（文档名 | 标题 | 相似度）
    print("--- 检索留痕 ---")
    for rank, (sim, idx) in enumerate(top_results, 1):
        chunk = chunks[idx]
        print(f"Top {rank} | {chunk['doc_name']} | {chunk['title']} | 相似度: {sim:.4f}")
    print("-" * 30)
        
    # 3. 拼接用户消息
    user_prompt = build_prompt(question, top_results)
    
    # 4. 组装 messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    # 5. 调用 API (temperature=0.1 减少幻觉)
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.1
        )
        answer = response.choices[0].message.content
        print("\n🤖 模型回答:\n" + answer)
        return answer, "ok"
    except Exception as e:
        print(f"\n❌ API 调用失败: {e}")
        return None, "api_error"