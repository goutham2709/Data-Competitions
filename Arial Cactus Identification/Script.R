library(keras)
library(tidyverse)

#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")

#BiocManager::install("EBImage")

library(EBImage)
library(tidyr)
library(future)
library(furrr)
library(caret)

train_path <- './train'
test_path <- './test'
height <- 32
width <- 32

labels <- read_csv('./train.csv')
test_ids <- data.frame(id = list.files(test_path)) %>% as_tibble()

preprocess_image <- function(file, path_, w, h){
  image <- readImage(paste(path_,file, sep = '/'), type="jpeg")                         
  image <- resize(image, w = w, h = h)  
  image <- clahe(image)                               
  image <- normalize(image)                                                            
  imageData(image)                                    
}

list2tensor <- function(xList) {
  xTensor <- simplify2array(xList)
  aperm(xTensor, c(4, 1, 2, 3))    
}

get_images <- function(data, path_, w, h){
  imgs <- data %>% select(id) %>% 
    mutate(ImageData = future_map(id, ~preprocess_image(.,path_ = path_, w = w, h = h))) %>%
    select(ImageData)
  return(list2tensor(imgs$ImageData))
}

set.seed(1234)
inTrain <- createDataPartition(labels$has_cactus, p = 0.9, list = FALSE)
train_data <- labels[inTrain,]
val_data <- labels[-inTrain,]
y_train <- train_data$has_cactus
y_val <- val_data$has_cactus

plan(multiprocess)
X_train <- get_images(data = train_data, path_ = train_path, w = width, h = height)
X_val <- get_images(data = val_data, path_ = train_path, w = width, h = height)
X_test <- get_images(data = test_ids, path_ = test_path, w = width, h = height)

model <- keras_model_sequential()

model %>%
  
  layer_conv_2d(
    filter = 32, kernel_size = c(3,3), padding = "same", 
    input_shape = c(32, 32, 3)
  ) %>%
  layer_activation("relu") %>%
  
  layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same") %>%
  layer_activation("relu") %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  layer_flatten() %>%
  layer_dense(512) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  
  layer_dense(1) %>%
  layer_activation("sigmoid")

opt <- optimizer_rmsprop(lr = 0.0001, decay = 1e-6)

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = opt,
  metrics = "accuracy"
)

set.seed(1234)
model %>% fit(X_train, y_train, epochs = 100, validation_data = list(X_val, y_val), batch_size = 32, verbose = 1)

y_test <- model %>% predict(X_test, batch_size = 32)

submission <- data.frame(id = test_ids$id, has_cactus = y_test)

write_csv(submission, 'submission.csv')