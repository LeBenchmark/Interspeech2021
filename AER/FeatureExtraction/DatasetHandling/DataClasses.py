
class Data():
    pass

class PersonalInfo(Data):
    def __init__(self):
        self.ID = ""
        self.age = ""
        self.gender = "U" # "M" for male, "F" for female or "U" as unknown
        self.language = "" # "French", "Chinese"

    def setParams(self, ID, age, gender, language):
        self.ID = ID
        self.age = age
        self.gender = gender
        self.language = language


class Annotation(Data):
    def __init__(self):
        self.ID = ""
        self.genre = "" # "Arousal", "Happiness", "Confidence"
        self.dimension = [] # [-1, 1] if having different length frames each having 1 value
        self.path = ""
        self.headers = [] # exp: ["GoldStandard"] or ["Arousal", "Valence"]
        self.annotator_info = PersonalInfo()

    def setParams(self, ID, genre, dimension, path, headers):
        self.ID = ID
        self.genre = genre
        self.dimension = dimension
        self.path = path
        self.headers = headers


class Features(Data):
    def __init__(self):
        self.ID = ""
        self.genre = "" # "eGeMAPS"
        self.dimension = [] # [-1, 39] if different length frames each having 39 values
        self.path = ""

    def setParams(self, ID, genre, dimension, path):
        self.ID = ID
        self.genre = genre
        self.dimension = dimension
        self.path = path


class AudioSample(Data):
    def __init__(self):
        self.ID = ""
        self.path = ""
        self.partition = "" # "train", "dev", "test"
        self.transcriptions = {} # {'ID':''}, where ID can be "Ziyi", "ASR", ...
        self.speaker_info = PersonalInfo()
        self.features = {} # [1, Features()]
        self.annotations = {} # [Annotation()]
    
    def setParams(self, ID, path, partition):
        self.ID = ID
        self.path = path
        self.partition = partition


def classToDic(dic):
    if isinstance(dic, dict): 
        for key in dic.keys():
            if isinstance(dic[key], Data):
                dic[key] = classToDic(dic[key])
            if isinstance(dic[key], list):
                for i, item in enumerate(dic[key]):
                    dic[key][i] = classToDic(item)
    elif isinstance(dic, Data):
        return classToDic(dic.__dict__)
    return dic


def localizePaths(dic, mainPath): # Change this path to that path
    if isinstance(dic, dict):
        # print("dict", dic)
        for key in dic.keys():
            if key == "path":
                dic[key] = dic[key].replace(mainPath, ".")
                dic[key] = dic[key].replace("./", "")
                if dic[key][0] == ".": dic[key] = dic[key][1:]
                # print(dic)
            else:
                localizePaths(dic[key], mainPath)
    return dic


def changePaths(dic, This, That): # Change this path to that path
    if isinstance(dic, dict):
        # print("dict", dic)
        for key in dic.keys():
            if key == "path":
                dic[key] = dic[key].replace(This, That)
                # print(dic)
            else:
                changePaths(dic[key], This, That)
    return dic
    # elif isinstance(dic, str):
    #     print(dic)
    #     if dic == "path":
            
    # else:
    #     return dic
    

# rater = PersonalInfo()
# rater.setParams("iop","23","F","French")
# annotation = Annotation()
# annotation.setParams("test1","Arousal",[-1,3],"somewhere",rater)
# print(annotation.annotator_info.language)

# sample = AudioSample()
# print(classToDic(sample.__dict__))

