require(MASS)

#Create an R function that can be used to clear the screen
clc <- function() cat(rep("\n",50))
clc()
# Number of random cases for Monte Carlo model
n = 200 
mu1 = c(-10, 0, 0, 0, 1, 1, 0, 0)
mu2 = c(10, 0, 0, 0, 1, 1, 0, 0)
mu3 = c(0, -10, 0, 0, 1, 1, 0, 0)
mu4 = c(0, 10, 0, 0, 1, 1, 0, 0)
mu5 = c(0, 0, -10, 0, 1, 1, 0, 0)
mu6 = c(0, 0, 10, 0, 1, 1, 0, 0)
mu7 = c(0, 0, 0, 0, 1, 1, 0, 0)
Sigma = diag(8)
# Usethis to make the randomly generated data the same ueach time you run the simulation.
# Change the 123459 to another large integer to chnage the sequence.
# Omit line to have a truly random sequence of numbers each time you run it.
set.seed(123459) 
g1<- mvrnorm(n, mu1, Sigma, tol = 1e-6)
g2<- mvrnorm(n, mu2, Sigma, tol = 1e-6)
g3<- mvrnorm(n, mu3, Sigma, tol = 1e-6)
g4<- mvrnorm(n, mu4, Sigma, tol = 1e-6)
g5<- mvrnorm(n, mu5, Sigma, tol = 1e-6)
g6<- mvrnorm(n, mu6, Sigma, tol = 1e-6)
g7<- mvrnorm(n, mu7, Sigma, tol = 1e-6)

m=as.matrix(rep(1,200))
g1=cbind(g1,m)
g2=cbind(g2, 2 * m)
g3=cbind(g3, 3 * m)
g4=cbind(g4, 4 * m)
g5=cbind(g5, 5 * m)
g6=cbind(g6, 6 * m)
g7=cbind(g7, 7 * m)

rs= rbind(g1,g2,g3,g4,g5,g6,g7)
write.table(rs, "/Users/shanshanli/Documents/LiClipse Workspace/kmeans/simulateData2.txt", sep = ",", row.names = FALSE, col.names = FALSE)

