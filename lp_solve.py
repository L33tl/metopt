import numpy as np

class CustomSimplexSolver:
    def __init__(self):
        self.objective_type = None
        self.c_original = None
        self.A_original = None
        self.b_original = None
        self.num_original_vars = 0
        self.constraint_types = None

        self.tableau = None
        self.basis_variable_indices = None

        self.slack_var_cols = []
        self.surplus_var_cols = []
        self.artificial_var_cols = []
        self.num_slack = 0
        self.num_surplus = 0
        self.num_artificial = 0
        self.num_tableau_vars = 0

    def read_problem_from_file(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        self.objective_type = lines[0].strip().upper()
        self.c_original = np.array(list(map(float, lines[1].strip().split())))
        self.num_original_vars = len(self.c_original)

        constraints_list = []
        b_list = []
        constraint_types = []

        i = 2
        while i < len(lines):
            constraint_line_parts = lines[i].strip().split()
            constraint_type = constraint_line_parts[0]
            rhs = float(constraint_line_parts[1])
            coeffs = np.array(list(map(float, lines[i+1].strip().split())))

            if len(coeffs) != self.num_original_vars:
                raise ValueError(f"Constraint coefficients line {i+2} does not match number of variables ({self.num_original_vars}).")

            constraints_list.append(coeffs)
            b_list.append(rhs)
            constraint_types.append(constraint_type)
            i += 2

        self.A_original = np.array(constraints_list)
        self.b_original = np.array(b_list)
        self.constraint_types = constraint_types


    def _standardize_problem(self):
        num_constraints = len(self.constraint_types)
        

        self.num_slack = self.constraint_types.count('<=')
        self.num_surplus = self.constraint_types.count('>=')

        self.num_artificial = self.constraint_types.count('=') + self.constraint_types.count('>=')
        
        self.num_tableau_vars = self.num_original_vars + self.num_slack + self.num_surplus + self.num_artificial
        
        num_tableau_rows = num_constraints + 2
        self.tableau = np.zeros((num_tableau_rows, self.num_tableau_vars + 1))
        
        self.basis_variable_indices = [0] * num_constraints

        current_slack_col = self.num_original_vars
        current_surplus_col = self.num_original_vars + self.num_slack
        current_artificial_col = self.num_original_vars + self.num_slack + self.num_surplus

        for r_idx in range(num_constraints):
            self.tableau[r_idx, :self.num_original_vars] = self.A_original[r_idx]
            self.tableau[r_idx, -1] = self.b_original[r_idx] # RHS


            if self.constraint_types[r_idx] == '<=':
                self.tableau[r_idx, current_slack_col] = 1
                self.basis_variable_indices[r_idx] = current_slack_col
                self.slack_var_cols.append(current_slack_col)
                current_slack_col += 1
            elif self.constraint_types[r_idx] == '=':
                self.tableau[r_idx, current_artificial_col] = 1
                self.basis_variable_indices[r_idx] = current_artificial_col
                self.artificial_var_cols.append(current_artificial_col)
                current_artificial_col += 1
            elif self.constraint_types[r_idx] == '>=':
                self.tableau[r_idx, current_surplus_col] = -1
                self.tableau[r_idx, current_artificial_col] = 1 # Artificial for >=
                self.basis_variable_indices[r_idx] = current_artificial_col
                self.surplus_var_cols.append(current_surplus_col)
                self.artificial_var_cols.append(current_artificial_col)
                current_surplus_col += 1
                current_artificial_col += 1

        z_phase1_row_idx = num_constraints
        
        for r_idx in range(num_constraints):
            basic_var_col = self.basis_variable_indices[r_idx]
            if basic_var_col in self.artificial_var_cols:
                self.tableau[z_phase1_row_idx, :] -= self.tableau[r_idx, :]
                
        z_phase2_row_idx = num_constraints + 1
        self.tableau[z_phase2_row_idx, :self.num_original_vars] = self.c_original
        
        if self.objective_type == 'MAX':
            self.tableau[z_phase2_row_idx, :self.num_tableau_vars] *= -1 

        for r_idx in range(num_constraints):
            basic_var_col = self.basis_variable_indices[r_idx]
            coeff_in_z_row = self.tableau[z_phase2_row_idx, basic_var_col]
            if abs(coeff_in_z_row) > 1e-9:
                self.tableau[z_phase2_row_idx, :] -= coeff_in_z_row * self.tableau[r_idx, :]

    def _pivot(self, pivot_row, pivot_col):
        pivot_element = self.tableau[pivot_row, pivot_col]
        
        if abs(pivot_element) < 1e-9:
            raise ValueError("Pivot element is zero or too close to zero.")

        self.tableau[pivot_row, :] /= pivot_element
        
        for r in range(self.tableau.shape[0]):
            if r != pivot_row:
                factor = self.tableau[r, pivot_col]
                self.tableau[r, :] -= factor * self.tableau[pivot_row, :]

        self.basis_variable_indices[pivot_row] = pivot_col


    def _find_entering_variable(self, z_row_index):
        z_row_coeffs = self.tableau[z_row_index, :-1]
        
        if np.all(z_row_coeffs >= -1e-9):
            return -1

        entering_col = np.argmin(z_row_coeffs)
        return entering_col


    def _find_leaving_variable(self, entering_col):
        ratios = []
        for r in range(len(self.basis_variable_indices)):
            rhs_val = self.tableau[r, -1]
            col_val = self.tableau[r, entering_col]
            
            if col_val > 1e-9:
                ratios.append(rhs_val / col_val)
            else:
                ratios.append(np.inf)

        if all(r == np.inf for r in ratios):
            return -1

        leaving_row = np.argmin(ratios)
        return leaving_row


    def solve(self):
        self._standardize_problem()
        
        num_constraints = len(self.constraint_types)
        

        print("\n--- Starting Phase 1 ---")
        z_phase1_row_idx = num_constraints
        
        iteration = 0
        while iteration < 100:
            
            if abs(self.tableau[z_phase1_row_idx, -1]) < 1e-9:
                all_artificial_non_basic_or_zero = True
                for basic_var_idx in self.basis_variable_indices:
                    if basic_var_idx in self.artificial_var_cols:
                        row_of_art_var = self.basis_variable_indices.index(basic_var_idx)
                        if self.tableau[row_of_art_var, -1] > 1e-9:
                             all_artificial_non_basic_or_zero = False
                             break
                if all_artificial_non_basic_or_zero:
                    print(f"Phase 1 - Iteration {iteration+1}: Objective (sum of artificials) is zero. Moving to Phase 2.")
                    break

            entering_col = self._find_entering_variable(z_phase1_row_idx)
            
            if entering_col == -1:
                if abs(self.tableau[z_phase1_row_idx, -1]) < 1e-9:
                    print(f"Phase 1 - Iteration {iteration+1}: Optimal, objective value is zero. Moving to Phase 2.")
                    break
                else:
                    return {"status": "NO_SOLUTION", "message": "Решений нет: Недопустимая область (минимальная сумма искусственных переменных > 0)."}

            leaving_row = self._find_leaving_variable(entering_col)
            if leaving_row == -1:
                return {"status": "UNBOUNDED_PHASE1", "message": "Phase 1: Проблема неограничена. (Обычно означает, что нет допустимого решения)"}
            
            self._pivot(leaving_row, entering_col)
            iteration += 1
        
        if iteration == 100:
            print("Phase 1 reached max iterations. Possible cycling or slow convergence.")
            if self.tableau[z_phase1_row_idx, -1] > 1e-9:
                 return {"status": "NO_SOLUTION", "message": "Решений нет: Недопустимая область (фаза 1 не достигла нуля после множества итераций)."}

        non_artificial_cols = [col for col in range(self.num_tableau_vars) if col not in self.artificial_var_cols]
        
        num_cols_phase2_tableau = len(non_artificial_cols) + 1
        tableau_phase2 = np.zeros((num_constraints + 1, num_cols_phase2_tableau))
        
        original_z_phase2_row_content = self.tableau[num_constraints + 1, :]
        
        for r_idx in range(num_constraints):
            tableau_phase2[r_idx, :-1] = self.tableau[r_idx, non_artificial_cols]
            tableau_phase2[r_idx, -1] = self.tableau[r_idx, -1]

        tableau_phase2[num_constraints, :-1] = original_z_phase2_row_content[non_artificial_cols]
        tableau_phase2[num_constraints, -1] = original_z_phase2_row_content[-1]
        
        self.tableau = tableau_phase2
        
        new_basis_variable_indices = []
        for old_idx in self.basis_variable_indices:
            if old_idx in non_artificial_cols:
                new_basis_variable_indices.append(non_artificial_cols.index(old_idx))
            else:
                print(f"Warning: Artificial variable {old_idx} was basic. Attempting to find new basic variable for row.")
                new_basis_variable_indices.append(-1)
        self.basis_variable_indices = new_basis_variable_indices
        self.num_tableau_vars = len(non_artificial_cols)

        z_phase2_row_idx = num_constraints
        for r_idx in range(len(self.basis_variable_indices)):
            basic_var_col = self.basis_variable_indices[r_idx]
            if basic_var_col != -1:
                coeff_in_z_row = self.tableau[z_phase2_row_idx, basic_var_col]
                if abs(coeff_in_z_row) > 1e-9:
                    self.tableau[z_phase2_row_idx, :] -= coeff_in_z_row * self.tableau[r_idx, :]

        print("\n--- Starting Phase 2 ---")
        
        iteration = 0
        while iteration < 100:
            entering_col = self._find_entering_variable(z_phase2_row_idx)
            
            if entering_col == -1:
                break
            
            leaving_row = self._find_leaving_variable(entering_col)
            if leaving_row == -1:
                return {"status": "UNBOUNDED", "message": "Решений нет: Целевая функция неограничена."}
            
            self._pivot(leaving_row, entering_col)
            iteration += 1

        if iteration == 100:
            print("Phase 2 reached max iterations. Possible cycling or slow convergence.")

        print("\n--- Phase 2 complete. Extracting solution ---")


        final_original_var_values = np.zeros(self.num_original_vars)
        
        for r_idx, basic_var_col_idx_new in enumerate(self.basis_variable_indices):
            if basic_var_col_idx_new != -1:
                if basic_var_col_idx_new < len(non_artificial_cols):
                    original_col_index = non_artificial_cols[basic_var_col_idx_new]
                    if original_col_index < self.num_original_vars:
                        final_original_var_values[original_col_index] = self.tableau[r_idx, -1]

        optimal_objective_value = np.dot(self.c_original, final_original_var_values)
        
        return {
            "status": "OPTIMAL",
            "x_optimal": final_original_var_values,
            "objective_value": optimal_objective_value
        }

if __name__ == "__main__":
    with open('zlp.txt', 'r', encoding='utf-8') as f:
        example_problem_content = f.read()
    with open("lp_problem.txt", "w") as f:
        f.write(example_problem_content)

    solver = CustomSimplexSolver()
    try:
        solver.read_problem_from_file("lp_problem.txt")
        solution = solver.solve()

        if solution["status"] == "OPTIMAL":
            print("\nЗадача успешно решена!")
            print(f"Оптимальная точка x: {solution['x_optimal']}")
            print(f"Значение целевой функции в оптимальной точке: {solution['objective_value']}")
        else:
            print(f"\nСтатус решения: {solution['status']}")
            print(solution['message'])

    except FileNotFoundError:
        print("Ошибка: файл 'lp_problem.txt' не найден.")
    except ValueError as e:
        print(f"Ошибка парсинга файла: {e}")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
