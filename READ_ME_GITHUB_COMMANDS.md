# It looks like your chat UI is glitching! 

Since you can't see my messages, I've written the commands here for you to copy and paste.

I noticed in your terminal output that you are currently on a branch called `shubhams-version`, which is why pushing to `main` said everything was up to date.

Ah, I understand! 

When you pulled his code, you started working on the `shubhams-version` branch. When you ran `git commit` earlier, all your new code was officially saved onto that `shubhams-version` branch. 

In Git, a "merge" is just taking the new commits from one branch (Shubham's) and applying them to another branch (Yours).

**However, there is an even easier way!** If you just want to take your exact current code and force it up to your `vedaanth-dev` branch on GitHub without dealing with local merges, just run this single command:

```bash
git push origin HEAD:vedaanth-dev
```

This tells Git: *"Take my current code (HEAD) and push it directly to the vedaanth-dev branch on GitHub."*
