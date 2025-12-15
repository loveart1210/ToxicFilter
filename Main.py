from Module import RegexModule
import json

aa = RegexModule.mainRun()

with open("Output/RegexModule.json", "w", encoding="utf-8") as f:
    json.dump(aa, f, ensure_ascii=False, indent=2)


