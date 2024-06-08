from kiwipiepy import Kiwi
import re
from jamo import h2j, j2hcj

kiwi = Kiwi()
KIWI_DICT = {
    '일반 명사': 'NNG', '고유 명사': 'NNP', '의존 명사': 'NNB', '수사': 'NR', '대명사': 'NP', '동사': 'VV', '형용사': 'VA',
    '보조 용언': 'VX', '긍정 지정사': 'VCP', '부정 지정사': 'VCN', '관형사': 'MM', '일반 부사': 'MAG', '접속 부사': 'MAJ',
    '감탄사': 'IC', '주격 조사': 'JKS', '보격 조사': 'JKC', '관형격 조사': 'JKG', '목적격 조사': 'JKO', '부사격 조사': 'JKB',
    '호격 조사': 'JKV', '인용격 조사': 'JKQ', '보조사': 'JX', '접속 조사': 'JC', '선어말 어미': 'EP', '종결 어미': 'EF',
    '연결 어미': 'EC', '명사형 전성 어미': 'ETN', '관형형 전성 어미': 'ETM', '체언 접두사': 'XPN', '명사 파생 접미사': 'XSN',
    '동사 파생 접미사': 'XSV', '형용사 파생 접미사': 'XSA', '어근': 'XR', '마침표, 물음표, 느낌표': 'SF', '줄임표 …': 'SE',
    '여는 괄호 (, [': 'SSO', '닫는 괄호 ), ]': 'SSC', '구분자 , · / :': 'SC', '기타 기호': 'SY', '외국어': 'SL',
    '한자': 'SH', '숫자': 'SN'
}

def no_commas(text):
    """쉼표 사용 안했는지 확인합니다."""
    if ',' in text:
        return False
    else:
        return True
    
def count_pos_under(text, search, threshold=8):
    """threshold 이상/이하의 품사를 가지는지 확인합니다."""
    count = 0

    search_pos = KIWI_DICT[search]
    split_sentences = kiwi.split_into_sents(text)

    for sentence in split_sentences:
        tokens = kiwi.tokenize(sentence.text)
        for token in tokens:
            if token.tag == search_pos: 
                count += 1
    if count <= threshold:
        return True
    else:
        return False
        
def count_pos_over(text, search, threshold=3):
    """threshold 이상/이하의 품사를 가지는지 확인합니다."""
    count = 0

    search_pos = KIWI_DICT[search]
    split_sentences = kiwi.split_into_sents(text)

    for sentence in split_sentences:
        tokens = kiwi.tokenize(sentence.text)
        for token in tokens:
            if token.tag == search_pos: 
                count += 1

    if count >= threshold:
        return True
    else:
        return False
    
def count_pos_non(text, search, threshold=0):
    """threshold 이상/이하의 품사를 가지는지 확인합니다."""
    count = 0

    search_pos = KIWI_DICT[search]
    split_sentences = kiwi.split_into_sents(text)

    for sentence in split_sentences:
        tokens = kiwi.tokenize(sentence.text)
        for token in tokens:
            if token.tag == search_pos: 
                count += 1

    if count == threshold:
        return True
    else:
        return False

def count_sentence(text, threshold=5):
    """threshold 이하의 문장 수를 가지는지 확인합니다."""
    split_sentences = kiwi.split_into_sents(text)
    if len(split_sentences) == threshold:
        return True
    else:
        return False
    
def split_paragraph(text):
    """텍스트를 문단 단위로 분리합니다."""
    paragraphs = text.split('\n\n')
    # 빈 문단을 제거합니다.
    paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]
    
    return paragraphs

def count_paragraph(text, threshold=2):
    """threshold 이하의 문단 수를 가지는지 확인합니다."""
    if len(split_paragraph(text)) <= threshold:
        return True
    
    return False

def count_repeat(text, threshold=2):
    """문장 내에 threshold 이하의 반복되는 단어를 가지는지 확인합니다."""
    nouns = ['NNG', 'NNP', 'NNB', 'NR', 'NP']

    split_sentences = kiwi.split_into_sents(text)
    
    for sentence in split_sentences:
        tokens = kiwi.tokenize(sentence.text)
        noun_counts = {}
        
        for token in tokens:
            if token.tag in nouns:
                if token.form not in noun_counts:
                    noun_counts[token.form] = 0
                noun_counts[token.form] += 1

    for noun, count in noun_counts.items():
        if count > threshold:
            return False
            
    return True
    

def extract_braced_strings(text):
    """중괄호 안의 문자열을 추출합니다."""
    pattern = r'\[([^\]]*)\]'

    matches = re.findall(pattern, text)
    return matches


def check_braced_strings(text):
    """중괄호 안의 문자열을 확인합니다."""
    matches = extract_braced_strings(text)
    for match in matches:
        tokens = kiwi.tokenize(match)

        for token in tokens:
            if token.tag == 'NNP':
                return True
            else:
                return False
            
    return True
    
def check_first_consonant(text,consonant='ㄱ'):
    text = re.sub(r"[^ㄱ-ㅣ가-힣]", "", text)
    words = [j2hcj(h2j(c)) for c in text]
    count = sum([1 for word in words if word[0] == consonant])
    
    return count
    
def check_middle_vowel(text,consonant='ㅏ'):
    text = re.sub(r"[^ㄱ-ㅣ가-힣]", "", text)
    words = [j2hcj(h2j(c)) for c in text]
    count = sum([1 for word in words if consonant in word])

    return count
    
def check_final_consonant(text,consonant='ㄹ'):
    text = re.sub(r"[^ㄱ-ㅣ가-힣]", "", text)
    words = [j2hcj(h2j(c)) for c in text]
    count = sum([1 for word in words if word[-1] == consonant])

    return count

def honorific_haeyo(text):
    text = re.sub(r"[^ㄱ-ㅣ가-힣]", "", text)
    split_sentences = kiwi.split_into_sents(text)
    if any([1 for sent in split_sentences if sent.text.strip()[-1] != '요']):
        return False
    else: 
        return True

def honorific_hao(text):
    text = re.sub(r"[^ㄱ-ㅣ가-힣]", "", text)
    split_sentences = kiwi.split_into_sents(text)
    if any([1 for sent in split_sentences if sent.text.strip()[-1] not in ['오','소']]):
        return False
    else: 
        return True