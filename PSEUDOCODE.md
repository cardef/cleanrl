\documentclass{article}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{amsmath}

\newcommand{\vect}[1]{\boldsymbol{#1}} % Uncomment for vector notation

\begin{document}

\begin{algorithm}
\caption{Hamilton-Jacobi-Bellman (HJB) inspired Continuous Control}
\label{alg:hjb_continuous}
\begin{algorithmic}[1]
\State \textbf{Input:} Environment $Env$, total timesteps $T_{total}$, learning rate $\alpha$, buffer size $B$,
    batch size $N$, exploration noise $\sigma$, learning start $T_{start}$, policy frequency $K$
\State \textbf{Initialize:} 
    \State $\quad$ Actor network $\pi_\theta$, Critic network $V_\phi$ with parameters $\theta$, $\phi$
    \State $\quad$ EMA versions $\pi_\theta^{EMA}$, $V_\phi^{EMA}$ with decay rate $\lambda$
    \State $\quad$ Dynamic model $f_\psi$, Reward model $r_\xi$ with parameters $\psi$, $\xi$
    \State $\quad$ Replay buffer $\mathcal{B}$ with capacity $B$
    \State $\quad$ Optimizers for all networks with learning rate $\alpha$

\For{$t = 1$ \textbf{to} $T_{total}$}
    \State \textbf{Collect Experience:}
    \State $\quad$ If $t < T_{start}$: Sample action $\vect{a}_t \sim \mathcal{U}(Env.action\_space)$
    \State $\quad$ Else: $\vect{a}_t = \pi_\theta^{EMA}(\vect{s}_t) + \mathcal{N}(0, \sigma)$
    \State $\quad$ Execute $\vect{a}_t$, observe $\vect{s}_{t+1}$, $r_t$, done
    \State $\quad$ Store transition $(\vect{s}_t, \vect{a}_t, r_t, \vect{s}_{t+1}, done)$ in $\mathcal{B}$

    \If{$t > T_{start}$}
        \State \textbf{Model Training:}
        \State $\quad$ Train $f_\psi$ on $\mathcal{B}$ until validation loss < threshold
        \State $\quad$ Train $r_\xi$ on $\mathcal{B}$ until validation loss < threshold
        
        \State \textbf{Policy Improvement:}
        \For{$k = 1$ \textbf{to} $K$}
            \State Sample batch $\{\vect{s}_i, \vect{a}_i, r_i, \vect{s}_{i+1}\}_{i=1}^N \sim \mathcal{B}$
            \State Compute value gradient $\nabla_{\vect{s}} V_\phi^{EMA}(\vect{s}_i)$
            \State Predict dynamics $\vect{f}_i = f_\psi(\vect{s}_i, \vect{a}_i)$
            \State Compute Hamiltonian $H_i = r_\xi(\vect{s}_i, \vect{a}_i) + \nabla_{\vect{s}} V_\phi^{EMA}(\vect{s}_i)^\top \vect{f}_i$
            \State Update $\theta$ to minimize $-\frac{1}{N}\sum_i H_i$
            \State Update EMA: $\pi_\theta^{EMA} \leftarrow \lambda\pi_\theta^{EMA} + (1-\lambda)\pi_\theta$
        \EndFor
        
        \State \textbf{Value Function Update:}
        \For{$k = 1$ \textbf{to} $K$}
            \State Sample batch $\{\vect{s}_i, \vect{a}_i, r_i, \vect{s}_{i+1}\}_{i=1}^N \sim \mathcal{B}$
            \State Compute value gradient $\nabla_{\vect{s}} V_\phi(\vect{s}_i)$
            \State Predict dynamics $\vect{f}_i = f_\psi(\vect{s}_i, \vect{a}_i)$
            \State Compute Hamiltonian $H_i = r_\xi(\vect{s}_i, \vect{a}_i) + \nabla_{\vect{s}} V_\phi(\vect{s}_i)^\top \vect{f}_i$
            \State Compute HJB residual $R_i = H_i - \frac{\log \gamma}{\Delta t} V_\phi(\vect{s}_i)$
            \State Update $\phi$ to minimize $\frac{1}{2N}\sum_i R_i^2$
            \State Update EMA: $V_\phi^{EMA} \leftarrow \lambda V_\phi^{EMA} + (1-\lambda)V_\phi$
        \EndFor
    \EndIf
\EndFor

\State \textbf{Return:} Optimized policy $\pi_\theta^{EMA}$
\end{algorithmic}
\end{algorithm}

\end{document}
