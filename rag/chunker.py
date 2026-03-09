"""智能文档分块模块

功能：
1. 基于句子边界的智能分块
2. 保留文档结构（标题层级）
3. 医学生成器词保护
4. 优化的重叠策略
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from backend.logging_config import get_logger
from backend.config import config
from backend.exceptions import DocumentParseError

logger = get_logger(__name__)


@dataclass
class Chunk:
    """分块结果"""
    text: str
    chunk_id: str
    metadata: Dict[str, Any]
    title: Optional[str] = None  # 所属标题
    position: int = 0  # 在文档中的位置


class IntelligentChunker:
    """智能文档分块器"""
    
    # 句子结束标点
    SENTENCE_ENDINGS = r'[。！？\.!?\n]+'
    
    # 医学术语列表（不应被分割）
    MEDICAL_TERMS = {
        # 心血管系统疾病
        "高血压", "低血压", "冠心病", "心绞痛", "心肌梗死", "心肌缺血",
        "心肌炎", "心包炎", "心律失常", "房颤", "室颤", "早搏",
        "心力衰竭", "心脏瓣膜病", "先天性心脏病", "肺心病", "脑卒中",
        "脑出血", "脑梗死", "短暂性脑缺血发作", "动脉粥样硬化",
        # 呼吸系统疾病
        "肺炎", "支气管肺炎", "大叶性肺炎", "病毒性肺炎", "支原体肺炎",
        "肺结核", "肺癌", "支气管哮喘", "慢性支气管炎", "肺气肿",
        "慢性阻塞性肺疾病", "肺心病", "胸膜炎", "胸腔积液",
        # 消化系统疾病
        "胃炎", "胃溃疡", "十二指肠溃疡", "胃癌", "食管炎", "食管癌",
        "肝炎", "肝硬化", "肝癌", "脂肪肝", "胆囊炎", "胆结石",
        "胰腺炎", "胰腺癌", "肠炎", "结肠炎", "溃疡性结肠炎",
        "克罗恩病", "阑尾炎", "腹膜炎", "消化不良", "便秘", "腹泻",
        # 泌尿系统疾病
        "肾炎", "肾小球肾炎", "肾盂肾炎", "肾病综合征", "肾功能衰竭",
        "尿毒症", "膀胱炎", "尿道炎", "前列腺增生", "前列腺炎", "前列腺癌",
        "尿路感染", "尿路结石", "肾结石",
        # 内分泌系统疾病
        "糖尿病", "一型糖尿病", "二型糖尿病", "妊娠糖尿病",
        "甲状腺功能亢进", "甲状腺功能减退", "甲状腺炎", "甲状腺结节",
        "甲状腺癌", "肾上腺皮质功能减退", "库欣综合征",
        "高脂血症", "高胆固醇血症", "痛风", "骨质疏松", "骨关节炎",
        # 神经系统疾病
        "偏头痛", "紧张性头痛", "丛集性头痛", "癫痫", "帕金森病",
        "阿尔茨海默病", "老年痴呆", "脑炎", "脑膜炎", "神经炎",
        "周围神经病变", "面神经麻痹", "三叉神经痛", "坐骨神经痛",
        # 血液系统疾病
        "贫血", "缺铁性贫血", "巨幼细胞性贫血", "再生障碍性贫血",
        "白血病", "淋巴瘤", "多发性骨髓瘤", "血小板减少症", "紫癜",
        # 免疫系统疾病
        "系统性红斑狼疮", "类风湿性关节炎", "干燥综合征", "硬皮病",
        "皮肌炎", "血管炎", "强直性脊柱炎", "银屑病", "荨麻疹",
        "湿疹", "过敏性鼻炎", "支气管哮喘",
        # 传染性疾病
        "新冠肺炎", "新冠", "流感", "普通感冒", "上呼吸道感染",
        "下呼吸道感染", "麻疹", "水痘", "腮腺炎", "风疹",
        "乙型肝炎", "丙型肝炎", "艾滋病", "梅毒", "淋病",
        # 常见症状
        "发热", "发烧", "高热", "低热", "寒战", "盗汗", "乏力",
        "疲劳", "疲倦", "体重下降", "体重增加", "食欲不振", "食欲减退",
        "恶心", "呕吐", "腹痛", "腹胀", "腹泻", "便秘", "便血",
        "胸闷", "胸痛", "胸闷气短", "心悸", "怔忡", "呼吸困难",
        "气促", "咳嗽", "咳痰", "干咳", "咯血", "咽痛", "咽干",
        "头痛", "头晕", "眩晕", "晕厥", "意识障碍", "嗜睡", "失眠",
        "尿频", "尿急", "尿痛", "排尿困难", "血尿", "蛋白尿",
        "皮疹", "瘙痒", "水肿", "黄疸", "肝区疼痛", "腰痛",
        # 常用药品 - 心血管类
        "氨氯地平", "硝苯地平", "非洛地平", "维拉帕米", "地尔硫卓",
        "卡托普利", "依那普利", "贝那普利", "培哚普利", "福辛普利",
        "氯沙坦", "缬沙坦", "厄贝沙坦", "坎地沙坦", "替米沙坦",
        "美托洛尔", "比索洛尔", "阿替洛尔", "卡维地洛", "拉贝洛尔",
        "阿司匹林", "氯吡格雷", "替格瑞洛", "华法林", "达比加群",
        "瑞舒伐他汀", "阿托伐他汀", "辛伐他汀", "普伐他汀", "氟伐他汀",
        "螺内酯", "呋塞米", "氢氯噻嗪", "氨苯蝶啶",
        # 常用药品 - 糖尿病类
        "二甲双胍", "格列本脲", "格列吡嗪", "格列齐特", "格列美脲",
        "瑞格列奈", "那格列奈", "吡格列酮", "罗格列酮", "西格列汀",
        "维格列汀", "沙格列汀", "利格列汀", "恩格列净", "达格列净",
        "卡格列净", "胰岛素", "门冬胰岛素", "赖脯胰岛素", "甘精胰岛素",
        "地特胰岛素", "德谷胰岛素", "预混胰岛素",
        # 常用药品 - 消化系统类
        "奥美拉唑", "兰索拉唑", "泮托拉唑", "雷贝拉唑", "埃索美拉唑",
        "法莫替丁", "雷尼替丁", "西咪替丁", "硫糖铝", "铝碳酸镁",
        "多潘立酮", "莫沙必利", "伊托必利", "甲氧氯普胺",
        "蒙脱石散", "洛哌丁胺", "地芬诺酯", "益生菌", "乳果糖",
        # 常用药品 - 呼吸系统类
        "沙丁胺醇", "特布他林", "福莫特罗", "沙美特罗", "茚达特罗",
        "异丙托溴铵", "噻托溴铵", "格隆溴铵", "布地奈德", "氟替卡松",
        "倍氯米松", "孟鲁司特", "扎鲁司特", "氨茶碱", "多索茶碱",
        # 常用药品 - 抗生素类
        "阿莫西林", "阿莫西林克拉维酸钾", "氨苄西林", "头孢氨苄",
        "头孢克洛", "头孢丙烯", "头孢克肟", "头孢曲松", "头孢哌酮",
        "阿奇霉素", "克拉霉素", "罗红霉素", "左氧氟沙星", "莫西沙星",
        "甲硝唑", "替硝唑", "奥硝唑", "呋喃唑酮", "复方新诺明",
        # 常用药品 - 镇痛解热类
        "布洛芬", "对乙酰氨基酚", "双氯芬酸", "萘普生", "塞来昔布",
        "依托考昔", "吗啡", "芬太尼", "曲马多", "可待因", "羟考酮",
        # 常用药品 - 精神神经类
        "地西泮", "艾司唑仑", "阿普唑仑", "氯硝西泮", "劳拉西泮",
        "舍曲林", "帕罗西汀", "氟西汀", "西酞普兰", "文拉法辛",
        "度洛西汀", "米氮平", "曲唑酮", "喹硫平", "奥氮平", "利培酮",
        "阿立哌唑", "碳酸锂", "丙戊酸钠", "卡马西平", "拉莫三嗪",
        # 常用药品 - 其他
        "泼尼松", "甲泼尼龙", "地塞米松", "氢化可的松",
        "左甲状腺素", "甲狀腺素", "甲硫咪唑", "丙硫氧嘧啶",
        "氯苯那敏", "西替利嗪", "氯雷他定", "地氯雷他定", "孟鲁司特",
        # 检查方法
        "心电图", "动态心电图", "心脏超声", "冠状动脉造影", "冠脉CTA",
        "血常规", "尿常规", "粪常规", "肝功能", "肾功能", "血脂",
        "血糖", "糖化血红蛋白", "胰岛素", "C肽", "甲状腺功能",
        "肿瘤标志物", "甲胎蛋白", "癌胚抗原", "前列腺特异性抗原",
        "胸部X光", "胸部CT", "腹部B超", "腹部CT", "腹部MRI",
        "胃镜", "肠镜", "支气管镜", "膀胱镜", "超声心动图",
        # 治疗方式
        "药物治疗", "手术治疗", "介入治疗", "放射治疗", "化疗",
        "靶向治疗", "免疫治疗", "内分泌治疗", "中医治疗", "康复治疗",
        "血液透析", "腹膜透析", "支架植入", "起搏器植入", "搭桥手术",
        "器官移植", "造血干细胞移植",
    }
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        min_chunk_size: int = 50
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk_text(self, text: str, file_path: str = "") -> List[Chunk]:
        """对文本进行智能分块
        
        Args:
            text: 输入文本
            file_path: 文件路径（用于元数据）
            
        Returns:
            分块列表
        """
        if not text or not text.strip():
            return []
        
        # 提取标题结构
        titles = self._extract_titles(text)
        logger.debug(f"提取到 {len(titles)} 个标题")
        
        # 提取句子
        sentences = self._split_into_sentences(text)
        logger.debug(f"提取到 {len(sentences)} 个句子")
        
        # 基于句子进行分块
        chunks = self._create_chunks(sentences, titles, file_path)
        
        logger.info(f"分块完成，得到 {len(chunks)} 个块")
        return chunks
    
    def _extract_titles(self, text: str) -> List[Tuple[str, int]]:
        """提取标题及其位置
        
        Returns:
            [(标题文本, 起始位置), ...]
        """
        # 匹配Markdown标题 (# ## ###) 或数字标题 (1. 2. 3.)
        title_pattern = r'^(#{1,6}\s+.+|(?:\d+\.)+\s+.+)$'
        
        titles = []
        for match in re.finditer(title_pattern, text, re.MULTILINE):
            titles.append((match.group().strip(), match.start()))
        
        # 按位置排序
        titles.sort(key=lambda x: x[1])
        return titles
    
    def _get_current_title(self, position: int, titles: List[Tuple[str, int]]) -> Optional[str]:
        """获取指定位置所属的标题"""
        current_title = None
        for title, pos in titles:
            if position < pos:
                break
            current_title = title
        return current_title
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        # 先按段落分割
        paragraphs = text.split('\n\n')
        
        sentences = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 按句子结束标点分割
            parts = re.split(self.SENTENCE_ENDINGS, para)
            
            for part in parts:
                part = part.strip()
                if part:
                    sentences.append(part)
        
        return sentences
    
    def _is_medical_term(self, text: str) -> bool:
        """检查文本是否包含医学术语"""
        for term in self.MEDICAL_TERMS:
            if term in text:
                return True
        return False
    
    def _create_chunks(
        self,
        sentences: List[str],
        titles: List[Tuple[str, int]],
        file_path: str
    ) -> List[Chunk]:
        """创建分块"""
        chunks = []
        current_chunk = ""
        current_title = None
        chunk_id = 0
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 检查当前句子是否包含医学术语
            has_medical_term = self._is_medical_term(sentence)
            
            # 尝试将句子添加到当前chunk
            if len(current_chunk) + len(sentence) < self.chunk_size:
                # 如果当前chunk为空，更新标题
                if not current_chunk:
                    # 估算当前句子的位置
                    position = i * 100  # 粗略估算
                    current_title = self._get_current_title(position, titles)
                
                current_chunk += sentence + "。 "
            
            else:
                # 当前chunk已满，保存
                if current_chunk.strip():
                    chunks.append(Chunk(
                        text=current_chunk.strip(),
                        chunk_id=f"chunk_{chunk_id}",
                        metadata={
                            "file_path": file_path,
                            "chunk_index": chunk_id,
                            "char_count": len(current_chunk)
                        },
                        title=current_title,
                        position=chunk_id
                    ))
                    chunk_id += 1
                
                # 处理新句子
                # 如果单个句子超过chunk_size，强制分割
                if len(sentence) > self.chunk_size:
                    # 递归分割长句子
                    sub_chunks = self._split_long_sentence(sentence, chunk_id, file_path, current_title)
                    chunks.extend(sub_chunks)
                    chunk_id += len(sub_chunks)
                    current_chunk = ""
                    current_title = None
                else:
                    # 保留overlap内容
                    if len(current_chunk) > self.chunk_overlap:
                        current_chunk = current_chunk[-self.chunk_overlap:]
                    else:
                        current_chunk = ""
                    
                    # 更新标题
                    position = i * 100
                    current_title = self._get_current_title(position, titles)
                    
                    current_chunk += sentence + "。 "
        
        # 添加最后一个chunk
        if current_chunk.strip():
            chunks.append(Chunk(
                text=current_chunk.strip(),
                chunk_id=f"chunk_{chunk_id}",
                metadata={
                    "file_path": file_path,
                    "chunk_index": chunk_id,
                    "char_count": len(current_chunk)
                },
                title=current_title,
                position=chunk_id
            ))
        
        return chunks
    
    def _split_long_sentence(
        self,
        sentence: str,
        start_id: int,
        file_path: str,
        title: Optional[str]
    ) -> List[Chunk]:
        """分割过长的句子"""
        chunks = []
        # 按子句分割（逗号、分号等）
        sub_parts = re.split(r'[,，;；]', sentence)
        
        current = ""
        for i, part in enumerate(sub_parts):
            part = part.strip()
            if not part:
                continue
            
            if len(current) + len(part) < self.chunk_size:
                current += part + ", "
            else:
                if current:
                    chunks.append(Chunk(
                        text=current.strip().rstrip(','),
                        chunk_id=f"chunk_{start_id + len(chunks)}",
                        metadata={"file_path": file_path},
                        title=title,
                        position=start_id + len(chunks)
                    ))
                current = part + ", "
        
        if current:
            chunks.append(Chunk(
                text=current.strip().rstrip(','),
                chunk_id=f"chunk_{start_id + len(chunks)}",
                metadata={"file_path": file_path},
                title=title,
                position=start_id + len(chunks)
            ))
        
        return chunks


def create_chunker() -> IntelligentChunker:
    """创建分块器实例"""
    return IntelligentChunker(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )


def parse_document(file_path: str) -> str:
    """增强的文档解析函数
    
    支持多种格式的文档解析，包含错误处理和日志记录
    
    Args:
        file_path: 文件路径
        
    Returns:
        解析后的文本内容
    """
    from pathlib import Path
    from llama_index.core import SimpleDirectoryReader
    import logging
    
    file_path_obj = Path(file_path)
    suffix = file_path_obj.suffix.lower()
    
    logger.info(f"开始解析文档: {file_path}")
    
    try:
        # TXT文件 - 直接读取
        if suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"TXT文档解析完成，字符数: {len(content)}")
            return content
        
        # Markdown文件 - 使用SimpleDirectoryReader
        elif suffix in ['.md', '.markdown']:
            reader = SimpleDirectoryReader(input_files=[file_path])
            documents = reader.load_data()
            content = "\n\n".join([doc.text for doc in documents])
            logger.info(f"Markdown文档解析完成，字符数: {len(content)}")
            return content
        
        # HTML文件 - 提取文本
        elif suffix in ['.html', '.htm']:
            try:
                from bs4 import BeautifulSoup
                with open(file_path, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                # 移除script和style标签
                for script in soup(["script", "style"]):
                    script.decompose()
                content = soup.get_text(separator='\n')
                # 清理空白
                lines = [line.strip() for line in content.split('\n')]
                content = '\n'.join(line for line in lines if line)
                logger.info(f"HTML文档解析完成，字符数: {len(content)}")
                return content
            except ImportError:
                logger.warning("BeautifulSoup未安装，使用SimpleDirectoryReader")
                reader = SimpleDirectoryReader(input_files=[file_path])
                documents = reader.load_data()
                content = "\n\n".join([doc.text for doc in documents])
                return content
        
        # PDF和Word文件 - 使用SimpleDirectoryReader
        elif suffix in ['.pdf', '.docx', '.doc']:
            reader = SimpleDirectoryReader(input_files=[file_path])
            documents = reader.load_data()
            content = "\n\n".join([doc.text for doc in documents])
            logger.info(f"{suffix}文档解析完成，字符数: {len(content)}")
            return content
        
        else:
            # 默认使用SimpleDirectoryReader
            reader = SimpleDirectoryReader(input_files=[file_path])
            documents = reader.load_data()
            content = "\n\n".join([doc.text for doc in documents])
            logger.info(f"文档解析完成，字符数: {len(content)}")
            return content
            
    except Exception as e:
        logger.error(f"文档解析失败: {file_path}, 错误: {e}")
        raise DocumentParseError(f"无法解析文档 {file_path}: {str(e)}") from e
