use core::slice;
use std::io::{self, Read, Write};


/**
 * Simple data format for fast reading of sets
 * with basic checks to avoid misuse.
 * 
 * Header
 * - 24-bit magic: E9, AA, 05
 * - 8-bit flags:
 *      LSB is 1 if datafile was written in little endian, 0 otherwise.
 * - u32 set count
 * 
 * Data
 * - array of set `length`s, each u32's
 * - array of sets of `length` items, where each element is an i32.
 */

const MAGIC: [u8; 3] = [0xe9, 0xaa, 0x05];
const LITTLE_ENDIAN_BIT: u8 = 1;

const MIN_SET_COUNT: u32 = 2;
const MAX_SET_COUNT: u32 = 256;

pub type DatafileSet = Vec<i32>;

#[derive(Debug)]
pub enum ReadError {
    Io(io::Error),
    BadMagic,
    BadEndianness,
    BadSetCount(u32),
}
#[derive(Debug)]
pub enum WriteError {
    Io(io::Error),
    BadSetCount(u32),
}

impl ToString for ReadError {
    fn to_string(&self) -> String {
        match self {
            ReadError::Io(e) => e.to_string(),
            ReadError::BadMagic => "bad magic".to_string(),
            ReadError::BadEndianness => {
                let expected = if little_endian() {
                    "little endian"
                } else {
                    "big endian"
                };
                format!("bad endianness - system is {}", expected)
            },
            ReadError::BadSetCount(c) =>
                format!("bad set count {}", c),
        }
    }
}

impl ToString for WriteError {
    fn to_string(&self) -> String {
        match self {
            WriteError::Io(e) => e.to_string(),
            WriteError::BadSetCount(c) =>
                format!("bad set count {}", c),
        }
    }
}

pub fn from_reader(mut reader: impl Read) -> Result<Vec<DatafileSet>, ReadError> {
    // Use unbuffered reading to avoid copying large sets.
    let header = {
        let mut header: [u8; 8] = [0; 8];
        reader.read_exact(&mut header)
            .map_err(|e| ReadError::Io(e))?;
        header
    };

    if header[0..3] != MAGIC {
        return Err(ReadError::BadMagic);
    }
    let le_bit_set = (header[3] & LITTLE_ENDIAN_BIT) != 0;
    if le_bit_set != little_endian() {
        return Err(ReadError::BadEndianness);
    }

    let set_count: u32 = unsafe { *(header.as_ptr().add(4) as *const u32) };
    if set_count < MIN_SET_COUNT || set_count > MAX_SET_COUNT {
        return Err(ReadError::BadSetCount(set_count));
    }

    let lengths = {
        let mut lengths: Vec<u32> = vec![0; set_count as usize];

        let lengths_slice = unsafe { slice::from_raw_parts_mut(
            lengths.as_mut_ptr() as *mut u8,
            set_count as usize * std::mem::size_of::<u32>()
        )};

        reader.read_exact(lengths_slice)
            .map_err(|e| ReadError::Io(e))?;

        lengths
    };

    let mut results: Vec<DatafileSet> = Vec::with_capacity(set_count as usize);

    for length in lengths {
        let mut result = vec![0; length as usize];
        
        let result_slice = unsafe { slice::from_raw_parts_mut(
            result.as_mut_ptr() as *mut u8,
            length as usize * std::mem::size_of::<i32>()
        )};

        reader.read_exact(result_slice)
            .map_err(|e| ReadError::Io(e))?;

        results.push(result);
    }

    Ok(results)
}

pub fn to_writer(mut writer: impl Write, sets: &[DatafileSet]) -> Result<(), WriteError> {
    // Use unbuffered writing to avoid copying large sets.
    let set_count = sets.len() as u32;
    if set_count < MIN_SET_COUNT || set_count > MAX_SET_COUNT {
        return Err(WriteError::BadSetCount(set_count));
    }

    let le_bit_set = if little_endian() { 1 } else { 0 };
    let count_slice: [u8; 4] = unsafe { std::mem::transmute(set_count) };

    let header: [u8; 8] = [
        MAGIC[0], MAGIC[1], MAGIC[2], le_bit_set,
        count_slice[0], count_slice[1], count_slice[2], count_slice[3]
    ];

    writer.write_all(&header)
        .map_err(|e| WriteError::Io(e))?;

    let lengths: Vec<u32> = sets.iter()
        .map(|s| s.len() as u32).collect();

    let lengths_slice = unsafe { slice::from_raw_parts(
        lengths.as_ptr() as *const u8,
        set_count as usize * std::mem::size_of::<u32>()
    )};

    writer.write_all(lengths_slice)
        .map_err(|e| WriteError::Io(e))?;

    for set in sets {
        let set_slice = unsafe { slice::from_raw_parts(
            set.as_ptr() as *const u8,
            set.len() * std::mem::size_of::<i32>()
        )};

        writer.write_all(set_slice)
            .map_err(|e| WriteError::Io(e))?;
    }
    Ok(())
}


#[cfg(target_endian = "little")]
const fn little_endian() -> bool {
    true
}

#[cfg(not(target_endian = "little"))]
const fn little_endian() -> bool {
    false
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pair() {
        test_write_read(&[
            vec![0, 4, 10, 20, 21, 26, 99],
            vec![0, 5, 6],
        ]);
    }

    #[test]
    fn test_kset() {
        test_write_read(&[
            vec![122, 120, 161, 155, 97, 86, 36, 32, 80, 9, 149, 140, 200, 82, 143, 30, 71, 33],
            vec![19, 6, 141, 90, 194, 167, 119, 125, 156, 197, 79, 98, 160, 28, 42, 111, 124, 18,],
            vec![74, 196, 139, 159, 78, 48, 168, 199, 169, 26, 15, 55, 182, 59, 10, 49, 165, 75,],
            vec![99, 190, 11, 107, 193, 69, 38, 137, 65, 44, 127, 62, 146, 135, 94, 50, 105, 123,],
            vec![41, 72, 68, 198, 195, 31, 134, 46, 103, 7, 61, 185, 188, 47, 152, 70, 189, 130],
            vec![126, 117, 148, 8, 178, 93, 166, 138, 171, 158, 131, 183, 157, 176, 16, 40, 34],
            vec![52, 54, 67, 122, 106, 12, 187, 150, 22, 172, 64, 163, 101, 37, 110, 170, 144],
            vec![113, 181, 87, 24, 60, 88, 164, 51, 63, 179, 43, 92, 186, 84, 25, 95, 115, 133],
            vec![85, 96, 132, 1, 56, 53, 89, 136, 21, 45, 27, 29, 58, 116, 2, 118, 39, 23, 20, 3],
            vec![100, 154, 4, 17, 129, 174, 57, 73, 145, 112, 14, 177, 184, 76, 91, 104, 83, 151],
            vec![173, 162, 81, 13, 114, 77, 66, 5, 191, 108, 180, 147, 175, 109, 35, 153, 128],
            vec![142, 192, 102]
        ]);
    }

    #[test]
    fn test_many_empty_sets() {
        test_write_read(&[vec![], vec![], vec![], vec![], vec![]]);
    }

    #[test]
    fn test_large_sets() {
        test_write_read(&[
            (0..(1<<16)-2).collect(),
            (0..(1<<17)+5).collect(),
            (0..(1<<14)/3).collect(),
        ]);
    }

    fn test_write_read(input: &[DatafileSet]) {
        let mut datafile: Vec<u8> = Vec::new();
        to_writer(&mut datafile, input).unwrap();

        let output = from_reader(datafile.as_slice()).unwrap();
        assert!(input == output);
    }
}
