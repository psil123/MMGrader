# from models import *
from prompt import *
from MentalModel import *
from models import *
import json
if __name__ == "__main__":
    model = Dummy()
    data=json.load(open('samples/sample3.json','r'))
    ob = MentalModel(data,"")
    ob.generate_MM()
    graph = ob.get_graph('abc123')
    ob.visualize_mental_model(graph)


