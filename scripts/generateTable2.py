import csv

dataset_list=[]
with open("./data_list","r") as whitelist:
    for line in whitelist.readlines():
        toRead=line.split(" ")[1]
        dataset=line.split(" ")[0]
        if(int(toRead)==1):
            dataset_list.append(dataset)
    whitelist.close()

output_folder="./outputs/"

nb_tries=3

for para in ["1threads"]:

    perfTable = [["Dataset", "FTM", "Vidal\\cite{vidal21}", "Ours $\\epsilon = 1\%$", "Ours $\\epsilon = 5\%$", "Ours $\\epsilon = 10\%$", "5\% speed up"]]

    speedUp_mean=0
    for i,dataset in enumerate(dataset_list):
        dataset=dataset.split(".")[0]
        print(dataset)
        if(dataset=="ctBones"):
            perfTable.append(["foot"])
        elif(dataset=="at"):
            perfTable.append(["AT"])
        elif(dataset=="ethanediol"):
            perfTable.append(["ethaneDiol"])
        else:
            perfTable.append([dataset])

        # FTM PD
        values=[]
        for itry in range(1,nb_tries+1):
            timeFTM=0
            filename = output_folder  +  "ftm_" + dataset + "_" + para + "_try_" + str(itry)

            with open(filename, "r") as f:
                for line in f.readlines():
                    line=line.split(" ")
                    if(len(line)>2):
                        if(line[1]=="\x1b[0mTotal"):
                            result=line[-1]
                            result=result.split("[")[1]
                            result=result.split("s")[0]
                            timeFTM+=float(result)
                f.close()
            values.append(timeFTM)

        values.sort()
        values = values[1:-1]
        mean = sum(values)/len(values)
        bestTime=mean
        perfTable[i+1].append(format(mean, '.2f'))

        # TVCG Prog
        values=[]
        for itry in range(1,nb_tries+1):
            time=0
            filename = output_folder  +  "tvcg_" + dataset + "_" + para + "_try_" + str(itry)

            with open(filename, "r") as f:
                for line in f.readlines():
                    line=line.split(" ")
                    if(len(line)>2):
                        if(line[1]=="\x1b[0mTotal"):
                            result=line[-1]
                            result=result.split("[")[1]
                            result=result.split("s")[0]
                            time=float(result)
                f.close()
            values.append(time)

        values.sort()
        values = values[1:-1]
        mean = sum(values)/len(values)
        perfTable[i+1].append(format(mean, '.2f'))
        bestTime = min(bestTime, mean)
        
        speedUp=-1
        for method in ["adaptive_uc1_"]:
        
            for eps in [0.01, 0.05, 0.1]:
                values=[]
                for itry in range(1,nb_tries+1):
                    time=0
                    filename = output_folder  +  method + dataset + "_eps" + str(eps) + "_" + para + "_try_" + str(itry)

                    if(dataset=="aneurism" and eps==0.05 and itry==1):
                        print(filename)
                    with open(filename, "r") as f:
                        for line in f.readlines():
                            line=line.split(" ")
                            if(len(line)>2):
                                if(line[1]=="\x1b[0mTotal" or line[1]=="\x1b[0mComplete"):
                                    if(line[2]!="memory"):
                                        result=line[-1]
                                        result=result.split("[")[1]
                                        result=result.split("s")[0]
                                        time+=float(result)
                                        if(dataset=="aneurism" and eps==0.05 and itry==1):
                                            print(method + " aneurism : " +str(result))
                        f.close()
                    values.append(time)

                values.sort()
                values = values[1:-1]
                mean = sum(values)/len(values)
                perfTable[i+1].append(format(mean, '.2f'))
                if(eps==0.05):
                    speedUp = mean/bestTime

        speedUp = 1-speedUp
        speedUp_mean+=speedUp
        perfTable[i+1].append(format(100*speedUp, '.1f')+"\%")


table = perfTable
print(perfTable)

with open("./table2.tex","w") as f:
    f.write("\\begin{tabular}{|l|rr|rrr|r|}\n")
    f.write("\hline\n")
    f.write("Dataset & TTK  & Vidal & \multicolumn{4}{c|}{Ours} \\\\ \n")
    f.write(" & & & $\\epsilon=1\\%$ & $\\epsilon=5\\%$ & $\\epsilon=10\\%$ & 5\\% speed up\\\\ \n")
    f.write("\hline\n")
    for row in table[1:]:
        f.write("%s\\\\\n" % " & ".join(col.title() for col in row))
    f.write("\hline\n")
    f.write("\\end{tabular}\n")

table[0] = ["Dataset", "TTK", "Vidal", "Ours epsilon = 1%", "Ours epsilon = 5%", "Ours epsilon = 10%", "5% speed up"]
with open("table2.csv", 'w', newline='') as file:
    mywriter = csv.writer(file, delimiter=',')
    mywriter.writerows(table)


