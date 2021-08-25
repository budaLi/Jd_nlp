# 你不知道学什么，往往是因为你学的太少   

# 2021.8.25  

人不能闲下来，闲着就会迷茫。时隔许久，尝试重新开始学习nlp,有时候觉得报班也挺好，人就是贱，付出了代价才知道努力。

# 2021.8.25 

## 002  训练营介绍，课程体系介绍

  介绍了项目开班，课程大概学习的内容

## 003 NLP定义及其歧义性

  1. NLP = NLU(语义理解) + NLG(语言生成)
  
## 004，005 机器翻译

   1. 统计机器翻译

![1.png](https://github.com/budaLi/jd_nlp/imgs/统计机器翻译_1.jpg)

    传统的机器翻译为;根据语料库里的单词与其翻译一一对应形成词库，翻译时根据对应的词进行直译。
    缺点： 速度慢、无语义分析、无上下文环境


   2. 中英文翻译

  今晚的课程有意思

      1) 分词： 今晚| 的| 课程| 有意思
      2) 直译   Tongith,of ,the course|interesting
      3) 将直译的单词排列组合，通过Language Model（语言模型），可以输出每一组排列组合对应的概率，即
          该模型可以判断输入的某一种排列组合更符合语法的概率，最高概率者即为翻译的结果。

  上述翻译的问题之一是，当翻译词汇过多时，排列组合的数量呈指数级，通过语言模型预测不太现实，时间复杂度为O（n**2）
  分词和翻译过程可以作为translation model,计算概率为langaage model,为了简化，是否可以将二者结合，提出Viterbi 算法。

   3.

       3.1 语言模型(language model)
           给定一句英文e,计算概率P(e)
           如果是符合英文语法的，p(e)高，如果是随机语句，p(e)低
       3.2 翻译模型（词典）
           给定一对<c,e>,计算p(c|e),c指的是中文，e指的是英文。
           语义相似度高则p(c|e)高，语义相似度低则p(c|e)低
       3.3 Decoding Algorithm（Viterbi）
           给定语言模型，翻译模型和f,找出最优的使得p(e)p(c|e)最大

   4. 语言模型

       语言模型是需要提前训练好的，对于一个好的语言模型，可以判断出句子是否符合语法，并给出概率：
       P(he is studing ai) > P(he studing is ai)

       也就是需要给出"he is studing ai"是句子的概率大于"he studing is ai"的概率，那么是如何计算的：
       Unigram: P(he is studing ai) = P(he) * P(is) * P(studing) * P(ai)   假设每个单词是独立的
       Bigram: (he is studing ai) = P(he) * P(is|he) * P(studing|is) * P(ai|studing)   假设当前单词只考虑与前一个单词相关
       Trigram: P(he is studing ai) = P(he) * p(is|he) * p(studing|he is) * P(ai| is studing)  假设当前单词与前两个单词相关
       N-gram        由Unigram、Bigram、Trigram可以延伸至N-gram,其中前三者是为了简化计算而假设得到的计算

       联合概率(joint probability)
       p(x1,x2） = p(x1) * p(x2|x1)  x1,x2的联合概率p(x1,x2) = 先验概率p(x1) * x1已知时x2的概率

           p(x1,x2,x3,x4)

          = p(x1)* p (x2|x1)* p(x3|x1,x2) *p(x4|x1,x2,x3)  # 为了简化，衍生出Unigram,Bigram,Trigram等

          = p(x1,x2) * p(x3|x1,x2) * p(x4|x1,x2,x3)

          = p(x1,x2,x3) * p(x4|x1,x2,x3)

          = p(x1,x2,x3,x4)

## 006 NLP项目实战

    1. 问答系统（ question answering)
    2. 情感分析（sentiment analysis)
       股票价格预测、舆情分析、产品评论、事件监测
    3. 机器翻译（machine translation)
    4. 自动摘要（text summarization)
    5. 聊天机器人(charbot) 闲聊形(seq2seq)、任务导向性(意图识别)
    6. 信息抽取（information extraction)


## 007 NLP关键技术

    Semantic(语义）
    Syntax(句子结构）
    Morphology(单词)
    Phonetics(声音)

    1. word segmentation(分词）
      今天是自然语言处理训练营第一次课
      今天 是  自然语言处理 训练营 第一次 课
    2. Part of Speech(词性）
       今天是1⽉22⽇，也是我们训练营的第⼀天，暂时课程，以ZOOM的⽅式直播
    3. Named Entity Recognition(命名实体识别)
       今天是（1⽉22⽇），也是我们(训练营)的第⼀天，暂时课程，以（ZOOM）的⽅式直播
    4. Parsing(句法分析）
    5. Dependency Parsing (依存分析）
    6. Relation Extraction(关系抽取）

### 008 时间复杂度

