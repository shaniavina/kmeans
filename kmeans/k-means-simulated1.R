require(MASS)

#Create an R function that can be used to clear the screen
clc <- function() cat(rep("\n",50))
clc()
# Number of random cases for Monte Carlo model
n1 = 100
n2 = 200
n3 = 300
n4 = 400
n5 = 500

mu1 = c(5,0,0)
mu2 = c(-5,0,0)
mu3 = c(0,5,0)
mu4 = c(0,-5,0)
mu5 = c(0,0,0)
Sigma = diag(3) * 0.3
# Usethis to make the randomly generated data the same ueach time you run the simulation.
# Change the 123459 to another large integer to chnage the sequence.
# Omit line to have a truly random sequence of numbers each time you run it.
set.seed(123459) 
g1<- mvrnorm(n1, mu1, Sigma, tol = 1e-6)
g2<- mvrnorm(n2, mu2, Sigma, tol = 1e-6)
g3<- mvrnorm(n3, mu3, Sigma, tol = 1e-6)
g4<- mvrnorm(n4, mu4, Sigma, tol = 1e-6)
g5<- mvrnorm(n5, mu5, Sigma, tol = 1e-6)

m1=as.matrix(rep(1,n1))
m2=as.matrix(rep(1,n2))
m3=as.matrix(rep(1,n3))
m4=as.matrix(rep(1,n4))
m5=as.matrix(rep(1,n5))

g1=cbind(g1,m1)
g2=cbind(g2, 2 * m2)
g3=cbind(g3, 3 * m3)
g4=cbind(g4, 4 * m4)
g5=cbind(g5, 5 * m5)

rs= rbind(g1,g2,g3,g4,g5)
write.table(rs, "/Users/shanshanli/Documents/LiClipse Workspace/kmeans/simulateData1(4).txt", sep = ",", row.names = FALSE, col.names = FALSE)

