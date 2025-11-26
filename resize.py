import csv

input_file = "./data/GlobalLandTemperaturesByCity.csv"
output_file = "data.csv"

keep_every = 10

with open(input_file, "r", newline='', encoding="utf-8") as f_in, \
     open(output_file, "w", newline='', encoding="utf-8") as f_out:

    reader = csv.reader(f_in)
    writer = csv.writer(f_out)

    for i, row in enumerate(reader):
        if i % keep_every == 0:
            writer.writerow(row)
