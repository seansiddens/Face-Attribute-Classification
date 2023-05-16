import csv

input_file = '../../celeba-dataset/list_attr_celeba.csv'
output_file = 'celeba_annotation.txt'

def main():
    # Read in the input file and extract the headers and data
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = [row for row in reader]

    # Extract the image filenames and labels from the data
    filenames = [row[0] for row in data]
    labels = [[int(val) for val in row[1:]] for row in data]

    # Write the output file
    with open(output_file, 'w') as f:
        # Write the number of images and number of classes as the first line
        num_images = len(filenames)
        num_classes = len(headers) - 1
        f.write(str(num_images) + '\n')
        f.write(' '.join(headers[1:]) + '\n')
        
        # Write the labels for each image
        for i in range(num_images):
            f.write(filenames[i] + ' ' + ' '.join(str(val) for val in labels[i]) + '\n')


if __name__ == "__main__":
    main()