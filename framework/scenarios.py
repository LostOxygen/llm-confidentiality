"""library implementing the different attack scenarios as enums"""
# pylint: disable=invalid-name, line-too-long
from enum import Enum


class Scenarios(Enum):
    """Scenario class providing the different scenarios as strings using Enums"""
    CloudPlain = 1
    CalendarPlain = 2
    MailPlain = 3
    NotesPlain = 4

    CalendarWithCloud = 5
    CalendarWithMail = 6
    CalendarWithNotes = 7
    CalendarWithCalendar = 8

    MailWithCloud = 9
    MailWithCalendar = 10
    MailWithNotes = 11
    MailWithMail = 12

    NotesWithCloud = 13
    NotesWithCalendar = 14
    NotesWithMail = 15
    NotesWithNotes = 16

    CloudWithCalendar = 17
    CloudWithMail = 18
    CloudWithNotes = 19
    CloudWithCloud = 20

    # possibly also website summary with tools combined
    # CloudPlain CalendarPlain MailPlain NotesPlain CalendarWithCloud CalendarWithMail CalendarWithNotes CalendarWithCalendar MailWithCloud MailWithCalendar MailWithNotes MailWithMail NotesWithCloud NotesWithCalendar NotesWithMail NotesWithNotes CloudWithCalendar CloudWithMail CloudWithNotes CloudWithCloud
