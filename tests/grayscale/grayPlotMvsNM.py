"""
NPD implementation

Copyright (C) 2023 Giuseppe Scarlato

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from tests.grayscale import grayPlotRRESSIM

def main(suffix, suffixNM, method):
    filenameList = [f'{method}_{suffix}', f'{method}_{suffixNM}']
    nameList = [f'{method}', f'{method} without momentum']
    saveNameList = [f'{method}', f'{method}_NM']
    grayPlotRRESSIM.main(filenameList, nameList, saveNameList=saveNameList)
