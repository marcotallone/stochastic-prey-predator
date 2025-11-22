$$
\begin{align*}
&\text{Set } \; T, \; x_0, \; N_{\text{paths}} & \\
&\text{errors\_list} \leftarrow [\;\;] & \\
& &\\
& \text{\textcolor{gray}{// Loop over different time steps $dt$}} & \\
&\text{for } k \text{ in } [3, 4, \dots, 15] \text{ do:} & \\
&\quad\quad N = 2^k & \\
&\quad\quad N_{\text{ref}} = 32 \cdot N & \\
&\quad\quad dt = T / N & \\
&\quad\quad dt_{\text{ref}} = T / N_{\text{ref}} & \\
&\quad\quad \text{RMSE\_list} \leftarrow [\;\;] & \\
& &\\
&\quad\quad \text{\textcolor{gray}{// Loop over different paths}} & \\
&\quad\quad \text{for } p \text{ in } [1, 2, \dots, N_{\text{paths}}] \text{ do:} & \\
&\quad\quad\quad\quad dW_{\text{ref}} \leftarrow \sqrt{dt_{\text{ref}}} \cdot \text{randn}(N_{\text{ref}}) \;\;\;\text{\textcolor{gray}{// generate $N_{\text{ref}}$ Gaussian vars}} & \\
&\quad\quad\quad\quad dW \leftarrow \text{sum}(\text{reshape}(dW_{\text{ref}}, 32, N), 1) \;\;\text{\textcolor{gray}{// sum $dW_{\text{ref}}$ every 32 steps}} & \\
&\quad\quad\quad\quad x_\text{ref} \leftarrow \text{simulate}(x_0, T, N_{\text{ref}}, dW_{\text{ref}}) & \\
&\quad\quad\quad\quad x \leftarrow \text{simulate}(x_0, T, N, dW) & \\
&\quad\quad\quad\quad \text{RMSE\_list}_p \leftarrow ||x - x_\text{ref}||_2 & \\
&\quad\quad \text{end for} & \\
& &\\
&\quad\quad \text{\textcolor{gray}{// Compute mean error for current $dt$}} & \\
&\quad\quad \text{errors\_list}_k \leftarrow \text{mean}(\text{RMSE\_list}) & \\
&\text{end for} & \\
& &\\
\end{align*}
$$


<!-- $$
\begin{align*}
&\text{Set } test & \\
&\text{Set } train & \\
&\quad\quad \text{losses\_list} \leftarrow [\;\;] & \\
& &\\
&\text{for } epoch \text{ in } [1, 2, \dots, N_{\text{epochs}}] \text{ do:} & \\
&\quad\quad \text{for } (x, y) \text{ in } \text{zip}(train.x, train.y) \text{ do:} & \\
\end{align*}
$$ -->