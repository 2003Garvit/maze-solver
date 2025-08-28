import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from collections import deque
import heapq

ROWS, COLS = 25, 25

def make_grid(r=ROWS, c=COLS):
    return [[1]*c for _ in range(r)]

def carve(grid, seed=None):
    if seed: random.seed(seed)
    r, c = len(grid), len(grid[0])
    stack = [(1,1)]; grid[1][1]=0
    while stack:
        x,y = stack[-1]
        nbrs=[]
        for dx,dy in [(2,0),(-2,0),(0,2),(0,-2)]:
            nx,ny=x+dx,y+dy
            if 0<nx<r-1 and 0<ny<c-1 and grid[nx][ny]==1:
                nbrs.append((nx,ny))
        if nbrs:
            nx,ny=random.choice(nbrs)
            grid[(x+nx)//2][(y+ny)//2]=0
            grid[nx][ny]=0
            stack.append((nx,ny))
        else: stack.pop()
    return grid

def neighbors(r,c,g):
    for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
        nr,nc=r+dr,c+dc
        if 0<=nr<ROWS and 0<=nc<COLS and g[nr][nc]==0: yield (nr,nc)

def bfs(grid,start,goal):
    q=deque([start]); came={start:None}; visited=[]
    while q:
        cur=q.popleft()
        visited.append(cur)
        if cur==goal: break
        for n in neighbors(*cur,grid):
            if n not in came: came[n]=cur; q.append(n)
    path=[]
    if goal in came:
        cur=goal
        while cur: path.append(cur); cur=came[cur]
        path.reverse()
    return path, visited

def dfs(grid,start,goal):
    stack=[start]; came={start:None}; visited=[]
    while stack:
        cur=stack.pop()
        visited.append(cur)
        if cur==goal: break
        for n in neighbors(*cur,grid):
            if n not in came: came[n]=cur; stack.append(n)
    path=[]
    if goal in came:
        cur=goal
        while cur: path.append(cur); cur=came[cur]
        path.reverse()
    return path, visited

def dijkstra(grid,start,goal):
    pq=[(0,start)]; dist={start:0}; came={start:None}; visited=[]
    while pq:
        d,cur=heapq.heappop(pq)
        visited.append(cur)
        if cur==goal: break
        for n in neighbors(*cur,grid):
            nd=d+1
            if n not in dist or nd<dist[n]: dist[n]=nd; came[n]=cur; heapq.heappush(pq,(nd,n))
    path=[]
    if goal in came:
        cur=goal
        while cur: path.append(cur); cur=came[cur]
        path.reverse()
    return path, visited

def astar(grid,start,goal):
    def h(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])
    pq=[(h(start,goal),0,start)]; gscore={start:0}; came={start:None}; visited=[]
    while pq:
        _,_,cur=heapq.heappop(pq)
        visited.append(cur)
        if cur==goal: break
        for n in neighbors(*cur,grid):
            ng=gscore[cur]+1
            if n not in gscore or ng<gscore[n]: gscore[n]=ng; came[n]=cur; heapq.heappush(pq,(ng+h(n,goal),ng,n))
    path=[]
    if goal in came:
        cur=goal
        while cur: path.append(cur); cur=came[cur]
        path.reverse()
    return path, visited

ALGO_MAP={"BFS":bfs,"DFS":dfs,"Dijkstra":dijkstra,"A*":astar}

def draw_maze(grid,path=[],visited_cells=[]):
    arr=np.array(grid)
    for r,c in visited_cells: arr[r][c]=2
    for r,c in path: arr[r][c]=3
    cmap = plt.get_cmap("viridis",4)
    plt.figure(figsize=(6,6))
    plt.imshow(arr, cmap=cmap, vmin=0, vmax=3)
    plt.axis("off")
    st.pyplot(plt)

st.title("Maze Solver")
algo = st.selectbox("Algorithm", ["BFS","DFS","Dijkstra","A*"])
seed = st.number_input("Seed (optional)", value=0)
speed = st.slider("Speed (ms per step)", 50, 500, 150)

if st.button("Generate and Solve Maze"):
    grid = carve(make_grid(), seed)
    start = (1,1)
    goal = (ROWS-2,COLS-2)
    st.write("Solving...")
    start_time = time.time()
    path, visited_cells = ALGO_MAP[algo](grid,start,goal)
    duration = time.time()-start_time
    st.write(f"Path length: {len(path)} | Visited cells: {len(visited_cells)} | Time: {duration:.3f}s")
    
    for i in range(0,len(visited_cells),1):
        draw_maze(grid, path=[], visited_cells=visited_cells[:i])
        time.sleep(speed/1000)
    draw_maze(grid, path=path, visited_cells=visited_cells)
    st.success("Maze Solved!")
