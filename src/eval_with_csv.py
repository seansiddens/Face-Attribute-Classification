import csv

ATTRIBUTES = "main_class sub_class img_id 5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young".split()

def convertToBinaryPred(): 
    return 0


def loadTruthLabels(): 
    return 0

def main(): 
    csvFilename = "prediction-v2.csv"
    outputFilename = "output.txt"
    truth = {"Blond_Hair": 221, "Male": -221} #input for truth labels
    pred = {} #hash with binary classifications
    accuracy = {}

    #load accuracy hash
    for k in truth.keys(): 
        accuracy[k] = 0 

    threshold = 0.5


    csvFile = open(csvFilename, "r")
    csvReader = csv.reader(csvFile)

    index = 0
    for row in csvReader: 
        #skip the header row
        if index == 0: 
            index += 1 
            continue  

        #populate each image_id with a zero vector 
        #assume each image_id is unique 
        n = len(row)
        pred[row[2]] = [0] * (n-3)
        
        #check each attribute 
        for i in range(3, 43): 
            if(float(row[i]) > threshold):
                pred[row[2]][i-3] = 1
            else: 
                pred[row[2]][i-3] = 0


    # keys = pred.keys()
    # for k in keys: 
    #     binaryPred = pred[k]
    #     n = len(ATTRIBUTES)     
    #     truthKeys = truth.keys()
    #     for i in range(n): 
    #         if ATTRIBUTES[i] in truthKeys: 
    #             print("here")
    #             key = ATTRIBUTES[i]
    #             accuracy[key] += binaryPred[i]

    #find the indexes 
    indexArray = []
    truthLabelArray = []
    truthLabels = truth.keys()
    for label in truthLabels: 
        index = ATTRIBUTES.index(label)
        indexArray.append(index - 3)
        truthLabelArray.append(label)

    #iterate through binary predictions
    keys = pred.keys()
    for k in keys: 
        binaryPred = pred[k]
        n = len(indexArray)
        for i in range(n): 
            index = indexArray[i]
            label = truthLabelArray[i]
            accuracy[label] += binaryPred[index]
            
    #output accuracy
    outputFile = open(outputFilename, "w")
    for label in truthLabelArray: 
        rawAcc = accuracy[label] / truth[label]
        roundedAcc = round(rawAcc, 2)
        outputString = f"Label: {label} had an accuracy of {roundedAcc}\n"
        outputFile.write(outputString)
        
            




  



    # with open(csvFilename, "r") as csvFile: 
    #     csvReader = csv.reader(csvFile)

    #     #convert output to binary classificaiton
    #     for row in csvReader:
    #         n = len(row)
    #         #set vector to 0 
    #         pred[row[2]] = [0] * (n-3)
    #         for i in range(3, n): 
    #             print(row[i][1])
    #         break



if __name__ == "__main__":
    main()