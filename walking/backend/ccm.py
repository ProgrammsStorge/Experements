import json
import random
import time
import traceback
from collections import defaultdict
import zstandard as zstd
from jinja2 import Template
from pathlib import Path

class CCM():
    def __init__(self,dataset=None,file=None,format="",zip=True,add_prefix=True,gram_count=5,repeat=False,save=True,to_set=True,hipo_model={}, hipo=[],hipo_multiple=1,no_setup=False):
        self.repeat=repeat
        cctx = zstd.ZstdCompressor(level=22)
        dctx = zstd.ZstdDecompressor()
        self.hipo=hipo
        self.attention_multiple=1.3
        self.hipo_multiple=hipo_multiple
        self.search_multiple=1.5
        self.hipo_model={}
        if dataset!=None:
            self.dataset=dataset
            self.wordset={}
            self.name=""
            self.gram_count=gram_count
            self.format=format
            s_time = time.time()
            print("Indexing...")
            prefix = self.indexing(self.dataset)
            suffix = f"{prefix // 1_000_000_000}B" if prefix >= 1_000_000_000 else f"{prefix // 1_000_000}M" if prefix >= 1_000_000 else f"{prefix // 1000}K" if prefix > 1000 else f"{prefix}"
            prefix = f"_{suffix}"
            if add_prefix:
                file+=prefix
            print(prefix)
            print("Time:", time.time() - s_time)

            print("Save...")
            original_text = json.dumps({"format": self.format, "gram_count": self.gram_count, "wordset": self.wordset})
            self.name=file
            self.model_name = Path(file).stem
            if zip:
                if save:
                    compressed_data = cctx.compress(original_text.encode('utf-8'))
                    with open(f"{file}.cws","wb") as f:
                        f.write(compressed_data)

            else:
                with open(f"{file}.json","w",encoding="utf-8") as f:
                    f.write(original_text)
        elif no_setup:
            pass
        else:
            with open(file,"rb") as f:
                if file.lower().endswith('.cws'):
                    readed = json.loads(dctx.decompress(f.read()).decode('utf-8'))
                else:
                    readed =json.loads(f.read())
                self.model_name = Path(file).stem
                self.wordset = readed["wordset"]
                self.to_set=to_set
                self.gram_count = readed["gram_count"]
                self.format = readed["format"]
                if readed.get("attention_multiple"):
                    self.attention_multiple =  readed["attention_multiple"]
                if readed.get("search_multiple"):
                    self.search_multiple =  readed["search_multiple"]
                if readed.get("hipo"):
                    self.hipo =  set([frozenset(context) for context in readed["hipo"]])
                if readed.get("hipo_model"):
                    self.hipo_model=readed["hipo_model"]
                if readed.get("hipo_multiple"):
                    self.hipo_multiple = readed["hipo_multiple"]

                if readed.get("memory")!=None:
                    self.memory = readed["memory"]
                else:
                    self.memory=False
                if hipo_model!={} and True:
                    self.wordset = self.merge_dicts(self.wordset,self.hipo_model)
                if to_set:
                    for word in self.wordset:
                        self.wordset[word] = set([frozenset(context) for context in self.wordset[word]])

    def chat(self,messages,max_tokens=250,end_tokens=[],ignore_tokens=[],load_function=None,print_debug=False,temperature=0,more_info=False,streaming=False):
        s_time = time.time()

        if self.memory==False:
            messages=[messages[-1]]
        elif isinstance(self.memory, (int, float, complex)):
            messages=messages[-self.memory:]

        template = Template(self.format)
        formatted = template.render(messages=messages).lower()
        attention=""
        if len(messages)>1:
            attention=template.render(messages=[messages[-1]]).lower()
        if streaming:
            to_ret = self.generate_streaming(formatted, max_tokens, end_tokens, ignore_tokens, load_function, print_debug,
                                   temperature, attention)
        else: to_ret=self.generate(formatted,max_tokens,end_tokens,ignore_tokens,load_function,print_debug,temperature,attention)
        #print("Time:", time.time() - s_time)
        if not more_info:return to_ret
        else: return to_ret,formatted+to_ret

    def merge_dicts(self,d1, d2):
        result = defaultdict(list)
        for d in (d1, d2):
            for k, v in d.items():
                if isinstance(v, list):
                    result[k].extend(v)
                else:
                    result[k] = v
        return dict(result)

    def extract_flat(self,wordset):
        unique_contexts = []
        seen = set()
        for word, contexts_list in wordset.items():
            for context_set in contexts_list:
                if isinstance(context_set, list):
                    key = frozenset(context_set)
                else:
                    key = frozenset(context_set)
                if key not in seen:
                    seen.add(key)
                    unique_contexts.append(context_set)
        return unique_contexts
    def generate(self,text_user,max_tokens=250,end_tokens=[],ignore_tokens=[],load_function=None,print_debug=False,temperature=0,attention=""):
        output_text=""
        gen = self.generate_streaming(text_user,max_tokens,end_tokens,ignore_tokens,load_function,print_debug,temperature,attention)
        for g in gen:
            output_text+=f"{g} "
        return output_text

    def generate_streaming(self,text_user,max_tokens=250,end_tokens=[],ignore_tokens=[],load_function=None,print_debug=False,temperature=0,attention=""):
        text=text_user
        internet_data={}
        internet_data_flat=[]
        if load_function!=None:
            inter_result = load_function(text_user)
            internet_data = self.indexing(inter_result,train=False)
            internet_data_flat=set([frozenset(context) for context in self.extract_flat(internet_data)])

           # print(internet_data,internet_data_flat)
            for word in internet_data:
                internet_data[word]=set([frozenset(context) for context in internet_data[word]])
            self.wordset=self.merge_dicts(self.wordset,internet_data)

        new_text=""
        for i in range(max_tokens):
            if not text or len(text) < self.gram_count:
                q_trigrams = {text} if text else set()
            else:
                q_trigrams = set(text[i:i + self.gram_count] for i in range(len(text) - 2))
            if not attention or len(attention) < self.gram_count:
                a_trigrams = {attention} if attention else set()
            else:
                a_trigrams = set(attention[i:i + self.gram_count] for i in range(len(attention) - 2))
            len_q = len(q_trigrams)
            word_win = ["", -999]
            last_winers=["","",""]
            if print_debug: print()
            try:
                for word in self.wordset:
                    for data in self.wordset[word]:
                            try:

                                intersection = len(q_trigrams & data)
                                union_len = len_q + len(data) - intersection
                                score = intersection / union_len if union_len else 0.0
                                if a_trigrams!=set():
                                   score *= (self.attention_multiple * len(a_trigrams & data))
                                if self.hipo_model!={} and data in self.hipo:
                                   score*=self.hipo_multiple
                                if internet_data!={} and data in internet_data_flat:
                                   score*=self.search_multiple
                                res = [[text, score]]
                                if temperature!=0: res[0][1] *= (random.uniform(-temperature, temperature))+1
                                if word in ignore_tokens:
                                    res[0][1] =0
                                if word_win[1] < res[0][1]:
                                    if print_debug: print("[", word, res[0][1], "]")
                                    word_win = [word, res[0][1]]
                                    last_winers.insert(0, word_win[0])
                                    last_winers.pop()
                                #if print_debug: print(f"\r{num_word}/{len(self.wordset)}", end="",flush=True)
                            except ValueError as e:
                                continue
            except Exception as e:
                if print_debug: traceback.print_exc()

            if not word_win[0] in new_text.split() or self.repeat:
                for token in end_tokens:
                    if token in word_win[0]:
                        if print_debug: print()
                        return new_text.strip()
                text += f" {word_win[0]}"
                new_text += f" {word_win[0]}"

                yield word_win[0]
                if print_debug: print(" score: "+str(word_win[1]), end="")
                if print_debug: print(new_text, end=" ")
            else:
                if print_debug: print()
                return new_text.strip()
        if print_debug: print()
        return new_text.strip()

    def indexing(self,dataset,train=True):
        edited_words = sorted(list(set(" ".join(dataset).split())))
        words_in_dataset = {}
        for num_word, word in enumerate(edited_words):
            for data in dataset:
                if word in data:
                    try:
                        data = " ".join(data.split()[:data.split().index(word)])
                        if words_in_dataset.get(word) == None:
                            words_in_dataset[word] = []
                        if len(data) < self.gram_count:
                            t_trigrams = {data} if data else set()
                        else:
                            t_trigrams = set(data[i:i + self.gram_count] for i in range(len(data) - 2))
                        words_in_dataset[word].append(list(t_trigrams))
                        if train: print(f"\r{num_word}/{len(edited_words)}", end="", flush=True)
                    except ValueError as e:
                        continue
        if train:
            self.wordset=words_in_dataset
            return sum(len(group) for contexts in self.wordset.values() for group in contexts)
        else:
            return words_in_dataset

class Pipeline():
    def __init__(self,file,code_execution=False,hipo_model={},to_set=True):
        self.file = file
        self.models={}
        self.code_execution=code_execution
        with open(file,'r',encoding="utf-8") as f:
            self.json_pipeline = json.load(f)
        for model in self.json_pipeline['models']:
            model_loaded=CCM(no_setup=True)
            readed = self.json_pipeline['models'][model]
            model_loaded.model_name = Path(file).stem
            model_loaded.wordset = readed["wordset"]
            model_loaded.to_set = to_set
            model_loaded.gram_count = readed["gram_count"]
            model_loaded.format = readed["format"]
            if readed.get("attention_multiple"):
                model_loaded.attention_multiple = readed["attention_multiple"]
            if readed.get("search_multiple"):
                model_loaded.search_multiple = readed["search_multiple"]
            if readed.get("hipo"):
                model_loaded.hipo = set([frozenset(context) for context in readed["hipo"]])
            if readed.get("hipo_model"):
                model_loaded.hipo_model = readed["hipo_model"]
            if readed.get("hipo_multiple"):
                model_loaded.hipo_multiple = readed["hipo_multiple"]
            if readed.get("memory") != None:
                model_loaded.memory = readed["memory"]
            else:
                model_loaded.memory = False
            if hipo_model != {} and True:
                model_loaded.wordset = model_loaded.merge_dicts(model_loaded.wordset, model_loaded.hipo_model)
            if to_set:
                for word in model_loaded.wordset:
                    model_loaded.wordset[word] = set([frozenset(context) for context in model_loaded.wordset[word]])
            self.models[model] = model_loaded
        self.pipeline_code=self.json_pipeline['code']
        if self.code_execution==True:
            self.ns = {
                "builtins": {},
                "models": self.models,
                "json_pipeline": self.json_pipeline,
            }
            exec(compile(self.pipeline_code, "<isolated>", "exec"), self.ns, self.ns)

    def chat(self,messages, *args, **kwargs):
        if "chat" not in self.ns:
            raise NameError("chat")
        return self.ns["chat"](messages,*args, **kwargs)