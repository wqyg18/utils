import numpy as np
import unittest

def cog2csg(cog_data_list):
    """
    将COG数据转换为CSG数据
    Parameters:
        cog_data_list: 包含8个接收器的COG数据列表,每个元素shape为(n_depths, n_times)
    Returns:
        csg_data_list: 包含n_depths个源位置的CSG数据列表,每个元素shape为(8, n_times)
    """
    n_receivers = len(cog_data_list)
    n_depths, n_times = cog_data_list[0].shape
    
    csg_data_list = []
    
    for s in range(n_depths):
        current_csg = np.zeros((n_receivers, n_times))
        for r in range(n_receivers):
            current_csg[r, :] = cog_data_list[r][s, :]
        csg_data_list.append(current_csg)
    
    return csg_data_list

def csg2cog(csg_data_list):
    """
    将CSG数据转换为COG数据
    Parameters:
        csg_data_list: 包含n_depths个源位置的CSG数据列表,每个元素shape为(8, n_times)
    Returns:
        cog_data_list: 包含8个接收器的COG数据列表,每个元素shape为(n_depths, n_times)
    """
    n_depths = len(csg_data_list)
    n_receivers, n_times = csg_data_list[0].shape
    
    cog_data_list = []
    
    for r in range(n_receivers):
        current_cog = np.zeros((n_depths, n_times))
        for s in range(n_depths):
            current_cog[s, :] = csg_data_list[s][r, :]
        cog_data_list.append(current_cog)
    
    return cog_data_list

class TestDataConversion(unittest.TestCase):
    def setUp(self):
        """生成测试数据"""
        # 设置随机种子以保证结果可重现
        np.random.seed(42)
        
        # 设置数据维度
        self.n_receivers = 8
        self.n_depths = 100
        self.n_times = 50
        
        # 生成随机COG数据
        self.original_cog_data = []
        for _ in range(self.n_receivers):
            cog = np.random.randn(self.n_depths, self.n_times)
            self.original_cog_data.append(cog)
    
    def test_cog_csg_conversion(self):
        """测试COG到CSG的转换,再转回COG是否与原始数据相同"""
        # COG -> CSG
        csg_data = cog2csg(self.original_cog_data)
        
        # CSG -> COG
        converted_cog_data = csg2cog(csg_data)
        
        # 验证数据维度
        self.assertEqual(len(converted_cog_data), self.n_receivers)
        self.assertEqual(converted_cog_data[0].shape, (self.n_depths, self.n_times))
        
        # 验证数据内容
        for i in range(self.n_receivers):
            self.assertTrue(np.allclose(converted_cog_data[i], self.original_cog_data[i]))
    
    def test_csg_shape(self):
        """测试CSG数据的形状是否正确"""
        csg_data = cog2csg(self.original_cog_data)
        
        # 验证CSG数据维度
        self.assertEqual(len(csg_data), self.n_depths)
        self.assertEqual(csg_data[0].shape, (self.n_receivers, self.n_times))

if __name__ == '__main__':
    unittest.main()
