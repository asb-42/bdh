import re

def cnt(t, pat):
    return len(re.findall(pat, t))

for f in ['xscript_zh_head.txt', 'lv_head.txt', 'ga_head.txt', 'xscript_iu.txt']:
    t = open(f, 'rb').read().decode('utf8', 'ignore')
    tot = max(1, len(t))
    parts = [('CJK', r'[\u4e00-\u9fff]'), ('kana', r'[\u3040-\u30ff]'), ('devanagari', r'[\u0900-\u097f]'),
             ('syllabics', r'[\u1400-\u167f]'), ('cyrillic', r'[\u0400-\u04ff]'), ('greek', r'[\u0370-\u03ff]'),
             ('latin', r'[A-Za-z]'), ('turkish-diacritic', r'[ığşöçüĞŞİ]')]
    row = "  ".join(f"{name} {100*cnt(t, pat)/tot:.2f}%" for name, pat in parts)
    print(f"{f:22s} chars={tot:8d}  {row}")

t = open('xscript_iu.txt', 'rb').read().decode('utf8', 'ignore')
lines = [l.strip() for l in t.split('\n') if len(l.strip()) > 2]
syl = [l for l in lines if re.search(r'[\u1400-\u167f]', l)]
tr = [l for l in lines if re.search(r'[ığşöçüĞŞİ]', l) and not re.search(r'[\u1400-\u167f]', l)]
en = [l for l in lines if not re.search(r'[\u1400-\u167fığşöçüĞŞİ]', l)]
print(f"\niu line classes: syllabic {len(syl)} ({sum(len(l.encode()) for l in syl)} B) | "
      f"turkish-only {len(tr)} ({sum(len(l.encode()) for l in tr)} B) | ascii-neither {len(en)} "
      f"({sum(len(l.encode()) for l in en)} B)")
print("first ascii-neither sample:", repr(en[0][:70]) if en else '-')
