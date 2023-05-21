import csv

ATTRIBUTES = "main_class sub_class img_id 5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young".split()
THRESHOLD = 0.5

def convertToBinaryPred(binaryPredHash, csvFilename): 
    csvFile = open(csvFilename, "r")
    csvReader = csv.reader(csvFile)

    imageNum = 0
    index = 0
    for row in csvReader: 
        #skip the header row
        if index == 0: 
            index += 1 
            continue  

        imageNum += 1 
        #populate each image_id with a zero vector 
        #assume each image_id is unique 
        n = len(row)
        binaryPredHash[row[2]] = [0] * (n-3)
        
        #check each attribute 
        for i in range(3, 43): 
            if(float(row[i]) > THRESHOLD):
                binaryPredHash[row[2]][i-3] = 1
            else: 
                binaryPredHash[row[2]][i-3] = 0

    return imageNum


def loadTruthLabels(truthLabels, accuracy, indexArray, truthLabelArray): 
    #load accuracy hash
    for k in truthLabels.keys(): 
        accuracy[k] = 0 

    #find the indexes 
    truthLabels = truthLabels.keys()
    for label in truthLabels: 
        index = ATTRIBUTES.index(label)
        indexArray.append(index - 3)
        truthLabelArray.append(label)

def calculateTotalLabels(pred, accuracy, indexArray, truthLabelArray): 
    keys = pred.keys()
    for k in keys: 
        binaryPred = pred[k]
        n = len(indexArray)
        for i in range(n): 
            index = indexArray[i]
            label = truthLabelArray[i]
            accuracy[label] += binaryPred[index]

def calculateAccuracy(truth, labelCount, numImages): 
    acc = {}
    k = truth.keys()
    for key in k: 
        pos = labelCount[key] / numImages 
        neg = (numImages - labelCount[key]) / numImages 
        acc[key] = (round(pos, 2), round(neg, 2))

    return acc 
def outputToFile(outputFilename, acc): 
    #output accuracy
    outputFile = open(outputFilename, "w")
    for key in acc.keys(): 
        pos = acc[key][0]
        neg = acc[key][1]
        outputString = f"Label: {key} Positive: {pos} Negative: {neg} \n"
        outputFile.write(outputString)

def main(csvFilename, outputFilename, truth): 
    pred = {} 
    labelCount = {}
    indexArray = []
    truthLabelArray = []
    imageNum = 0
    acc = {}

    loadTruthLabels(truth, labelCount, indexArray, truthLabelArray) 
    imageNum = convertToBinaryPred(pred, csvFilename)
    calculateTotalLabels(pred, labelCount, indexArray, truthLabelArray)
    acc = calculateAccuracy(truth, labelCount, imageNum)
    print(labelCount)
    print(acc)
    outputToFile(outputFilename, acc)

    

if __name__ == "__main__":
    #INPUTS TO SCRIPT: 
    csvFilename = "prediction-v2.csv"
    outputFilename = "blonde-women.txt"
    truth = {"Blond_Hair": 221, "Male": 0}
    #----------------------------------------

    main(csvFilename, outputFilename, truth)