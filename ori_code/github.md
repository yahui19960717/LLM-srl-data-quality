###### 提交的过程

1. 查看当前的状态
git status

2. 添加所有更改到暂存区
git add .
git add -A
git add --all

3. 提交更改
git commit -m "提交描述"

4. 推送
git push -u origin main
git push

5. 需要填写姓名和密码，密码就是17863807812@163.com
classic pat:用于push github
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

6. 我设置了git config --global credential.helper store，希望下次再输入一次姓名和密码之后不需要再输入了[已经可以了]


7. .gitignore只能忽略 还没有被git跟踪的文件/目录，如果文件之前已经git跟踪了，后面即使在.gitigore里面加了这一行，它也不会自动失效，git仍然会继续跟踪它。
 使用 git rm -r --cached ori_code/annotation/selected_data，停止跟踪，不删除本地文件
 然后在.gitignore里面再写上相关的文件路径，就可以忽略了

其他github备份：
PAT github
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
