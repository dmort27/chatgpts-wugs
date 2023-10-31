GERMAN_PREFIX_LONG = '''Fülle die Lücke mit dem korrekten Plural des Nomens "{}" aus. Antworte mit einem Wort.'''
GERMAN_SUFFIX_LONG = '''Hier ist {}. Jetzt sind es zwei ___!\n___ :'''
GERMAN_PREFIX_SHORT = '''Bilde den korrekten Plural des Nomens "{}". Antworte mit einem Wort.'''
GERMAN_SUFFIX_SHORT = '''{} :'''
GERMAN_SHOTS = [
    ("ein Fisch", "Fische"),
    ("eine Tür", "Türen"),
    ("ein Kind", "Kinder"),
    ("ein Park", "Parks"),
    ("ein Fenster", "Fenster"),
    ("eine Tochter", "Töchter"),
    ("ein Floß", "Flöße"),
    ("ein Wald", "Wälder")
]

ENGLISH_PREFIX_LONG = '''Fill in the blank with the correct past tense of the verb "{}". Answer with one word.'''
ENGLISH_SUFFIX_LONG = '''They {} all the time. In fact, they ___ just yesterday!\n___ :'''
ENGLISH_PREFIX_SHORT = '''Form the correct past tense of the verb "{}". Answer with one word.'''
ENGLISH_SUFFIX_SHORT = '''{} :'''
ENGLISH_SHOTS = [
    ("test", "tested"),
    ("teach", "taught"),
    ("build", "built"),
    ("sing", "sang"),
    ("hit", "hit")
]

TAMIL_PREFIX_LONG = '''"{}" என்ற வார்த்தையின் சரியான கடந்த காலத்துடன் காலி இடத்தை நிரப்பவும். ஒரு வார்த்தை கொடுங்கள்.'''
TAMIL_SUFFIX_LONG = '''நேற்று அவரிடம், "நீ {}" என்றேன். அதைக் கேட்டு அவன் போய் ___.\n___ :'''
TAMIL_PREFIX_SHORT = '''"{}" என்ற வார்த்தையின் கடந்த காலத்தை கொடுங்கள். ஒரு வார்த்தை கொடுங்கள்.'''
TAMIL_SUFFIX_SHORT = '''{} :'''
TAMIL_SHOTS = [
    ("கற்றுக்கொள்", "கற்றுக்கொண்டான்"),
    ("உட்காரு", "உட்கார்ந்தான்"),
    ("ஓடு", "ஓடினான்"),
    ("சாப்பிடு", "சாப்பிட்டான்"),
    ("வில்", "விற்றான்"),
    ("கிடை", "கிடைத்தான்"),
    ("இழு", "இழுத்தான்"),
    ("இழு", "இழுத்தான்")
]

TURKISH_PREFIX_LONG_1 = '''Boşlukları "{}" ile verilen eylemin birinci tekil şahıs geçmiş zaman formları ile doldurun.'''
TURKISH_SUFFIX_LONG_1 = '''Ben her zaman {}. Ama dün ___.\n___:'''
TURKISH_PREFIX_SHORT_1 = '''Tek bir sözcük ile farazi "{}" eyleminin birinci tekil şahıs geçmiş zaman hali nasıl olur?'''
TURKISH_SUFFIX_SHORT_1 = '''{} :'''
TURKISH_SHOTS_1 = [
    ("yerim", "yedim"),
    ("okurum", "okudum"),
    ("üterim", "üttüm"),
    ("uçarım", "uçtum"),
    ("öğürürüm", "öğürdüm")
]

TURKISH_PREFIX_LONG_2 = '''Boşlukları "{}" ile verilen eylemin ikinci çoğul şahıs hikaye geçmiş zaman negatif formları ile doldurun.'''
TURKISH_SUFFIX_LONG_2 = '''Siz her zaman {}. Ama dün ___.\n___:'''
TURKISH_PREFIX_SHORT_2 = '''Tek bir sözcük ile farazi "{}" eyleminin ikinci çoğul şahıs hikaye geçmiş zaman negatif formu nasıl olur?'''
TURKISH_SUFFIX_SHORT_2 = '''{} :'''
TURKISH_SHOTS_2 = [
    ("yersiniz", "yememişsiniz"),
    ("okursunuz", "okumamışsınız"),
    ("ütersiniz", "ütmemişsiniz"),
    ("uçarsınız", "uçmamışınız"),
    ("öğürürsünüz", "öğürmemişsiniz")
]

TURKISH_PREFIX_LONG_3 = '''Boşlukları "{}" ile verilen adın birinci tekil şahıs iyelik ve yönelme hali formları ile doldurun.'''
TURKISH_SUFFIX_LONG_3 = '''Dün {} geldik. Bugün de tekrar ___ gideceğiz.\n___:'''
TURKISH_PREFIX_SHORT_3 = '''Tek bir sözcük ile farazi "{}" adının birinci tekil şahıs iyelikli yönelme hali nasıl olur?'''
TURKISH_SUFFIX_SHORT_3 = '''{} :'''
TURKISH_SHOTS_3 = [
    ("okuldan", "okuluma"),
    ("evden", "evime"),
    ("teşekkülden", "teşekkülüme"),
    ("dedemden", "dedeme"),
    ("arkadaştan", "arkadaşıma")
]

TURKISH_PREFIX_LONG_4 = '''Boşlukları "{}" ile verilen adın belirtme hali formları ile doldurun.'''
TURKISH_SUFFIX_LONG_4 = '''Dün bir {} gördük. Bugün de tekrar o ___ göreceğiz.\n___:'''
TURKISH_PREFIX_SHORT_4 = '''Tek bir sözcük ile farazi {} adının belirtme hali nasıl olur?'''
TURKISH_SUFFIX_SHORT_4 = '''{} :'''
TURKISH_SHOTS_4 = [
    ("okul", "okulu"),
    ("ev", "evi"),
    ("teşekkül", "teşekkülü"),
    ("adam", "adamı"),
    ("çocuk", "çocuğu")
]


class PromptGenerator:
    def __init__(self, prefix, suffix, language, sep="\n\n"):
        # Prefix and suffix for all prompts
        self.prefix = prefix
        self.suffix = suffix
        # Carrier prompt for shots (for one-shot and few-shot learning)
        self.shot_prompt = self.suffix + " {}"
        # Language
        self.language = language
        # Separator token
        self.sep = sep
    
    def generate_prompt(self, word_suffix, shots=[]):
        if self.language == "german":
            word_prefix = word_suffix.split()[1]
        else:
            word_prefix = word_suffix
        if len(shots) == 0:
            prompt = (
                self.prefix + 
                self.sep + 
                self.suffix
            )
        else:
            prompt = (
                self.prefix + 
                self.sep + 
                self.sep.join([self.shot_prompt.format(*shot) for shot in shots]) +
                self.sep +
                self.suffix
            )
        return prompt.format(word_prefix, word_suffix)
    

def load_prompt(language, prompt_type="short"):
    if language == "german":
        if prompt_type == "short":
            return GERMAN_PREFIX_SHORT, GERMAN_SUFFIX_SHORT, GERMAN_SHOTS
        else:
            return GERMAN_PREFIX_LONG, GERMAN_SUFFIX_LONG, GERMAN_SHOTS
    elif language == "english":
        if prompt_type == "short":
            return ENGLISH_PREFIX_SHORT, ENGLISH_SUFFIX_SHORT, ENGLISH_SHOTS
        else:
            return ENGLISH_PREFIX_LONG, ENGLISH_SUFFIX_LONG, ENGLISH_SHOTS
    elif language == "tamil":
        if prompt_type == "short":
            return TAMIL_PREFIX_SHORT, TAMIL_SUFFIX_SHORT, TAMIL_SHOTS
        else:
            return TAMIL_PREFIX_LONG, TAMIL_SUFFIX_LONG, TAMIL_SHOTS
    elif language == "turkish_1":
        if prompt_type == "short":
            return TURKISH_PREFIX_SHORT_1, TURKISH_SUFFIX_SHORT_1, TURKISH_SHOTS_1
        else:
            return TURKISH_PREFIX_LONG_1, TURKISH_SUFFIX_LONG_1, TURKISH_SHOTS_1
    elif language == "turkish_2":
        if prompt_type == "short":
            return TURKISH_PREFIX_SHORT_2, TURKISH_SUFFIX_SHORT_2, TURKISH_SHOTS_2
        else:
            return TURKISH_PREFIX_LONG_2, TURKISH_SUFFIX_LONG_2, TURKISH_SHOTS_2
    elif language == "turkish_3":
        if prompt_type == "short":
            return TURKISH_PREFIX_SHORT_3, TURKISH_SUFFIX_SHORT_3, TURKISH_SHOTS_3
        else:
            return TURKISH_PREFIX_LONG_3, TURKISH_SUFFIX_LONG_3, TURKISH_SHOTS_3
    elif language == "turkish_4":
        if prompt_type == "short":
            return TURKISH_PREFIX_SHORT_4, TURKISH_SUFFIX_SHORT_4, TURKISH_SHOTS_4
        else:
            return TURKISH_PREFIX_LONG_4, TURKISH_SUFFIX_LONG_4, TURKISH_SHOTS_4
