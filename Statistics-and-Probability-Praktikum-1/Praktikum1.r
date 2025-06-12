data <- read.csv("nutrient.csv")


mode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

print("Mean")
print(sapply(data, mean))

print("Median")
print(sapply(data, median))

print("Mode")
print(sapply(data, mode))