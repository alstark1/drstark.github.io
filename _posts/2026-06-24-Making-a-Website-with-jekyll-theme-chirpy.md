---
title: Making a Website with jekyll-theme-chirpy (Beginner Level)
date: 2026-06-24 13:00:00 -0400
categories: [Website]
tags: [ruby]
math: true
---

# jekyll-theme-chirpy Tutorial
## Motivation
As a scientist, it is important to communicate effectively with the public and get people excited about your science/projects.
In the modern age websites are a popular form of sharing information, as such, it is useful to know how to make them (for free) and learn how they are structured.
The goal of this post is to guide you through obtaining the jekyll-theme-chirpy source code, setting up the proper environment, and deploying the website using GitHub.
## 1. Obtaining Source Code
There are 2 options to obtain source code. You can obtain it from the original GitHub page:

[https://github.com/cotes2020/jekyll-theme-chirpy](https://github.com/cotes2020/jekyll-theme-chirpy){:target="_blank" rel="noopener noreferrer"}

This tutorial may become out of date for the original source code as it is updated. 
I will be sticking with the version I obtained in February 2026; changes have been made to the functionality of the page to suit my preferences.
If issues are encountered with the original source code, you can copy my branch:

[https://github.com/alstark1/drstark.github.io](https://github.com/alstark1/drstark.github.io){:target="_blank" rel="noopener noreferrer"}
## 2. Setting up Environment and Deploying Locally 
Only Ruby is necessary to compile the website; however, an older version is required, namely 3.3.10.
If on a Windows machine, first download a Linux environment, here I will show how to install Windows Subsystem for Linux (WSL).
Simply run the following command in PowerShell:
```bash
wsl --install
```
Further information to customize your WSL version and linux distribution are available [here](https://learn.microsoft.com/en-us/windows/wsl/install){:target="_blank" rel="noopener noreferrer"}.
A new application named Terminal should be present on your machine.
Now, using Terminal on either macOS, Linux, or Windows, ensure that [Homebrew](https://brew.sh/){:target="_blank" rel="noopener noreferrer"} is installed by entering:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Read the brew output carefully, as you will need to execute some commands to make sure Homebrew is active in the current and future instances of your terminal.
Once Homebrew is installed, you should be able to use the brew command to install rbenv, which will control which version of Ruby your machine is using.
```bash
brew install rbenv ruby-build
```
ruby-build here is a dependency of rbenv and helps to install specified versions on your machine. 
rbenv then manages the versions you have installed, and you can specify which version you want used globally or in specific directories. 
For jekyll-theme-chirpy, we can run the following commands inside the jekyll-theme-chirpy (website) directory:
```bash
rbenv install 3.3.10
rbenv local 3.3.10
```
These commands will use ruby-build to install version 3.3.10 and rbenv to reference and use ruby 3.3.10 when compiling the website. 
Finally, you can run the following commands inside the jekyll-theme-chirpy (website) directory to build the website and deploy it locally before pushing any changes to GitHub:
```bash
bundle install
bundle exec jekyll serve
```
## 3. Customizations to Website Layout
Since I mainly want this page to showcase the knowledge and skills I have obtained, rather than the original home page, which displays the most recent posts.
I redirect the home page to an about page where I introduce myself.
The home page is contained in the index.html file in the jekyll-theme-chirpy (website) directory; it initially contains the following text:
```html
---
layout: home
# Index page
---
```
The index.html file is changed to:
```html
---
layout: null
permalink: /
# Index page
---

<meta http-equiv="refresh" content="0; url={{ site.baseurl }}/about/">
```
The layout in jekyll-theme-chirpy controls what is displayed when one clicks a tab on the sidebar (categories, tags, etc.).
With the layout null, no presets are loaded, and instead the permalink option and the final line allow for redirection to the about page.
The permalink sets the URL where the file is displayed; in this case the root of the site.
The final line will redirect the client to the about page when attempting to access the root/home page.
I also did not want the Archive tab on the sidebar; for this, I just created a directory named no_use_tabs in the website directory and moved the archives.md file from the _tabs directory into the no_use_tabs directory.
You can also set the order of the tabs and the associated icons in the front matter (between the --- lines) for the markdown files which specify the tab:
```markdown
---
layout: categories
icon: fas fa-stream
order: 4
---
```
A list of available icons can be found through [FontAwesome](https://fontawesome.com/icons){:target="_blank" rel="noopener noreferrer"}.
The _config.yml file is where you can modify much of the information on the page, such as the description and links to other websites (LinkedIn, GitHub, email, etc.) in the sidebar; this file is well commented and relatively self-explanatory.
An avatar picture can be set for your sidebar in the _config.yml file; to do this, place an image in /assets/img/image.jpg (or any image file type supported by the HTML &lt;img&gt; tag) and specify the path in the _config.yml file.
If the image is not behaving correctly, custom CSS assets can also be included in assets/css/file.css and then call this custom file from the _config.yml file.

Finally, I have changed the favicon (the icon which appears in the tab on your browser) to be a picture of the Kohn-Sham potential from my research.
This is relatively simple: first, crop an image so that it is square; in my case, I cropped to 96x96 pixels.
Next, insert this image into a favicon generator at [realfavicongenerator.net](https://realfavicongenerator.net/){:target="_blank" rel="noopener noreferrer"}, and it will return a zipped folder.
Insert the contents of this folder into a folder called favicons with the following file path /assets/img/favicons; Jekyll should take care of the rest.
There are other more subtle changes I have made here, and I will certainly make more in the future; if you want to see every detail, you can check out my [GitHub repository](https://github.com/alstark1/drstark.github.io){:target="_blank" rel="noopener noreferrer"}.
## 4. Deploying Through GitHub
To deploy your website through GitHub first create a repository and push your local website directory to the new repository.
To create a repository, navigate to the repositories tab on GitHub using the user navigation menu by clicking on your profile picture in the top right of the page.
Select the "New" option to the right of the "Find a repository..." search bar; you can set the options for your repository as follows (ensure that the repository is named &lt;username&gt;.github.io):

![Repository Settings](/assets/img/2026-06-24-website_post/rep_settings_1.png)

Next, push your local directory with the following commands while in the local website directory:
```bash
git init
git remote add origin https://github.com/<github-username>/<github-username>.github.io.git
git add .
git commit -m "message describing what you will push"
git branch -M main
git push -u origin main
```
The git init command creates a directory named .git in the website directory, allowing it to connect to the GitHub repository.
The second line connects the local repository to the one hosted on GitHub.
Next, the file changes can be staged and committed in the third and fourth lines; git add stages the files you want to include when modifying your branch, and git commit creates a snapshot of the staged changes.
The fifth line sets the local repository to track and make changes in the main branch.
Finally, git push pushs the changes from the git commit snapshot to the main branch of your GitHub repository.
The files and directories in the local repository should now appear in your GitHub repository, if this is not the case, do not continue until the local and GitHub repositories match.
Next, navigate to your website's GitHub repository in the browser and select settings in the bar at the top[1], then the pages tab on the sidebar[2].
Set the source in build and deployment to GitHub Actions[3].

![GitHub Actions Workflow](/assets/img/2026-06-24-website_post/actions_workflow_2.png)

After you click this, a box should appear prompting you to configure Jekyll; press the configure button[4].
If you have obtained the code from my repository, it should already include a jekyll.yml file and start building automatically.
If it prompts you to configure Jekyll, you can ignore steps 4-6.

![Configure Jekyll in Workflow](/assets/img/2026-06-24-website_post/config_jekyll_3.png)

A jekyll.yml file will be created in the .github/workflows directory after clicking configure. Click "Commit changes..."[5] to make the box appear, then click "Commit changes"[6].

![Commit Jekyll File into Workflow](/assets/img/2026-06-24-website_post/commit_jekyll_4.png)

Once this is finished, navigate to the Settings > Pages section of the GitHub repository again; it should indicate that the website is live at the URL[7].

![Page Posted](/assets/img/2026-06-24-website_post/page_posted_5.png)

Congratulations!!!

You have created a free website and can now communicate your work to the world.
I encourage you to customize the website further to suit your use case and aesthetic.
There are also other templates which use Jekyll and could be more aligned with your stylistic or functional preferences.
