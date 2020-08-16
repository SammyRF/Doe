import json

class JsonHandler:
    def __init__(self, path):
        if not path:
            path = "UserDefines.json"
        if path.endswith('json'):
            with open(path, "r") as read_file:
                self.data = json.load(read_file)
        else:
            self.data = json.loads(path)
        self.cases = {}
        self.__getCases_()

    def __getCases_(self):
        for case in self.data["cases"]:
            self.cases[case["casename"]] = case

    def getCase(self, casename):
        return self.cases[casename]
