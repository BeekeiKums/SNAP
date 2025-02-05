// ...existing code...
function displayTable(data) {
    const table = document.createElement('table');
    table.border = 1;
    const headerRow = document.createElement('tr');
    const selectAllTh = document.createElement('th');
    const selectAllCheckbox = document.createElement('input');
    selectAllCheckbox.type = 'checkbox';
    selectAllCheckbox.addEventListener('change', function() {
        const checkboxes = document.querySelectorAll('.row-checkbox');
        checkboxes.forEach(checkbox => {
            checkbox.checked = selectAllCheckbox.checked;
        });
    });
    selectAllTh.appendChild(selectAllCheckbox);
    headerRow.appendChild(selectAllTh);

    Object.keys(data[0]).forEach(key => {
        const th = document.createElement('th');
        th.textContent = key;
        th.dataset.column = key;
        th.style.display = columnVisibility[key] ? '' : 'none';
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);

    data.forEach((row, rowIndex) => {
        const tr = document.createElement('tr');
        const checkboxTd = document.createElement('td');
        const rowCheckbox = document.createElement('input');
        rowCheckbox.type = 'checkbox';
        rowCheckbox.className = 'row-checkbox';
        rowCheckbox.dataset.rowIndex = rowIndex;
        checkboxTd.appendChild(rowCheckbox);
        tr.appendChild(checkboxTd);

        Object.keys(row).forEach((key, index) => {
            const td = document.createElement('td');
            td.textContent = row[key];
            td.dataset.column = key;
            td.style.display = columnVisibility[key] ? '' : 'none';
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });

    const csvTable = document.getElementById('csvTable');
    csvTable.innerHTML = '';
    csvTable.appendChild(table);
}
// ...existing code...
