library(tidyverse)
library(rvest)


# read html of the site 


html <- read_html("https://editorial.rottentomatoes.com/guide/best-netflix-movies-to-watch-right-now/")

# We want to scrape 
#name of movie, Year of Movie,Ranking, Director of the movie, Tomato Score


#movie name

movie_name = html %>% html_elements(".article_movie_title a") %>% html_text()

#year

year = html %>% html_elements(".subtle.start-year") %>% html_text()
# refine year from (yyyy) to yyyy as numeric

year = substr(year,2,5)
year = as.numeric(year)

# ranking

ranking = html %>% html_elements(".countdown-index-resposive") %>% html_text()

ranking = substring(ranking,2)
ranking = as.numeric(ranking)


# directors

divs = html %>% html_elements(".info.director")
# managing multiple director to same movie
all_dirs <- sapply(divs, function(div) div%>% html_elements("a") %>% html_text())
all_dirs <- sapply(all_dirs, paste, collapse = ", ")

# tomato percent

tmeter <- html %>% html_elements(".tMeterScore") %>% html_text()
tmeter <- as.numeric(gsub("%","",tmeter)) # refining tmeter from "yy%" to yy


# Creating a data frame of Top movies
Top_movies <- data.frame(
  "Ranking" = ranking,
  "Name" = movie_name,
  "Year" = year,
  "Tomato_Score" = tmeter,
  "Director" = all_dirs
)


